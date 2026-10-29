import os
import re
import json
import uuid
import sqlite3
import hashlib
import datetime
import traceback
import subprocess
import threading
import shutil
from contextlib import closing
from pathlib import Path

import numpy as np
import requests
import torch

import gradio as gr
import chromadb

from chromadb.config import Settings

import docx2txt
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer, CrossEncoder

from kokoro import KPipeline


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_IMPL"] = (
    "chromadb.telemetry.dummy.DummyTelemetry"
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:8b")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "llama3.2:1b")
PODCAST_MODEL = os.getenv("PODCAST_MODEL", "qwen3:8b")
CHAT_NUM_CTX = int(os.getenv("CHAT_NUM_CTX", "8192"))
KOKORO_DEVICE = os.getenv("KOKORO_DEVICE", "cuda")
VOICE_A = os.getenv("KOKORO_VOICE_A", "pm_alex")
VOICE_B = os.getenv("KOKORO_VOICE_B", "pf_dora")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

DOCUMENTS_PATH = "/app/data/documentos"
AUDIO_PATH = "/app/audio/podcasts"
INDEX_DB_PATH = os.getenv("INDEX_DB_PATH", "/app/indice/rag.sqlite3")

# Coleção com os fragmentos filhos (vetores). Os blocos pais ficam no SQLite.
COLLECTION_NAME = "rag_filhos"
# Coleções de versões anteriores, removidas na próxima indexação para liberar espaço
LEGACY_COLLECTIONS = ["edital_avancado"]

# O e5-instruct exige instrução nas consultas (os documentos vão sem prefixo)
E5_QUERY_PREFIX = (
    "Instruct: Given a question, retrieve passages from documents that answer the question\n"
    "Query: "
)

SAMPLE_RATE = 24000
CHROMA_BATCH = 1000
SEPARADOR_FONTES = "\n\n---\n📚 **Fontes:**"

# ============================================================
# AUTENTICAÇÃO
# ============================================================

AUTH_USER = os.getenv("APP_USER", "beta")
AUTH_PASS = os.getenv("APP_PASS", "teste123")

# ============================================================
# CRIA DIRETÓRIOS
# ============================================================

os.makedirs(DOCUMENTS_PATH, exist_ok=True)
os.makedirs(AUDIO_PATH, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_DB_PATH), exist_ok=True)

# ============================================================
# CONTROLE DE GPU
# ============================================================

GPU_DISPONIVEL = torch.cuda.is_available()


def resolver_device(valor):
    """Usa CUDA só se pedido e realmente disponível."""
    return "cuda" if (valor == "cuda" and GPU_DISPONIVEL) else "cpu"


DEVICE_PADRAO = "cuda" if GPU_DISPONIVEL else "cpu"
EMBED_DEVICE = resolver_device(os.getenv("EMBED_DEVICE", DEVICE_PADRAO))
RERANK_DEVICE = resolver_device(os.getenv("RERANK_DEVICE", DEVICE_PADRAO))
TTS_DEVICE = resolver_device(KOKORO_DEVICE)

print("=" * 70)
print("SISTEMA RAG + PODCAST")
print("=" * 70)
print(f"Ollama:              {OLLAMA_URL}")
print(f"Chat model:          {CHAT_MODEL}")
print(f"Podcast LLM:         {PODCAST_MODEL}")
print(f"Embedding model:     {EMBEDDING_MODEL}")
print(f"Reranker model:      {RERANKER_MODEL}")
print(f"Embedding device:    {EMBED_DEVICE.upper()}")
print(f"Reranker device:     {RERANK_DEVICE.upper()}")
print(f"TTS device:          {TTS_DEVICE}")
print(f"Voice A:             {VOICE_A}")
print(f"Voice B:             {VOICE_B}")
print("=" * 70)

# ============================================================
# MODELO DE EMBEDDING (uma única instância, fp16 na GPU)
# ============================================================

print(f"Carregando embedding ({EMBED_DEVICE.upper()})...")

# Na GPU carrega direto em fp16: metade da VRAM, sem reservar a cópia fp32
DTYPE_GPU = {"torch_dtype": torch.float16}

embedder = SentenceTransformer(
    EMBEDDING_MODEL,
    device=EMBED_DEVICE,
    model_kwargs=DTYPE_GPU if EMBED_DEVICE == "cuda" else None
)

print("Embedding pronto.")


def liberar_vram():
    """Devolve ao driver a memória que o PyTorch mantém reservada após uso pesado."""
    if GPU_DISPONIVEL:
        torch.cuda.empty_cache()


def embed_documentos(textos):
    vetores = embedder.encode(
        textos,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return vetores.astype(np.float32).tolist()


def embed_pergunta(pergunta):
    return embed_documentos([E5_QUERY_PREFIX + pergunta])[0]


# ============================================================
# RERANKER
# ============================================================

print(f"Carregando reranker ({RERANK_DEVICE.upper()})...")

# max_length limita o custo: um bloco pai (~2500 caracteres) cabe em ~700 tokens
reranker = CrossEncoder(
    RERANKER_MODEL,
    device=RERANK_DEVICE,
    max_length=1024,
    automodel_args=DTYPE_GPU if RERANK_DEVICE == "cuda" else None
)

liberar_vram()

print("Reranker pronto.")

# ============================================================
# ESTADO GLOBAL DE MODELOS (modificável em runtime)
# ============================================================

# Esses valores são lidos pelas funções de chat/debate
# e podem ser alterados pela aba Configurações sem reiniciar
_config = {
    "chat_model":    CHAT_MODEL,
    "podcast_model": PODCAST_MODEL,
    "summary_model": SUMMARY_MODEL,
    "voice_a":       VOICE_A,
    "voice_b":       VOICE_B,
}

VOZES_KOKORO = {
    "pm_alex":  "👨 Alex (masculino PT)",
    "pm_santa": "👨 Santa (masculino PT)",
    "pf_dora":  "👩 Dora (feminino PT)",
}

# Formato (rótulo, valor) aceito pelos dropdowns do Gradio
OPCOES_VOZES = [(rotulo, voz) for voz, rotulo in VOZES_KOKORO.items()]


def listar_modelos_ollama():
    """
    Consulta a API do Ollama e retorna lista de modelos baixados.
    Retorna lista de strings com os nomes dos modelos.
    """
    try:
        resp = requests.get(OLLAMA_URL + "/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        modelos = [m["name"] for m in data.get("models", [])]
        return sorted(modelos) if modelos else ["(nenhum modelo encontrado)"]
    except Exception as e:
        print(f"Erro ao listar modelos Ollama: {e}")
        return ["(erro ao conectar ao Ollama)"]


def recarregar_e_atualizar_dropdowns():
    """
    Busca modelos atuais do Ollama e atualiza
    os três dropdowns de modelo.
    """
    modelos = listar_modelos_ollama()
    chat    = _config["chat_model"]
    podcast = _config["podcast_model"]
    summary = _config["summary_model"]

    # Se o modelo configurado não estiver na lista, adiciona para não quebrar
    for m in [chat, podcast, summary]:
        if m not in modelos and not modelos[0].startswith("("):
            modelos.append(m)

    return (
        gr.Dropdown(choices=modelos, value=chat),
        gr.Dropdown(choices=modelos, value=podcast),
        gr.Dropdown(choices=modelos, value=summary),
        "✅ Lista de modelos atualizada.\n\nModelos disponíveis:\n"
        + "\n".join(f"  • {m}" for m in modelos)
    )


def salvar_configuracoes(chat_m, podcast_m, summary_m, va, vb):
    """
    Aplica as configurações selecionadas ao estado global.
    Não reinicia o container — efeito imediato na próxima chamada.
    As vozes também passam a ser o padrão da aba Debate.
    """
    _config["chat_model"]    = chat_m
    _config["podcast_model"] = podcast_m
    _config["summary_model"] = summary_m
    _config["voice_a"]       = va
    _config["voice_b"]       = vb

    print(f"[Config] chat={chat_m} | podcast={podcast_m} | summary={summary_m}")
    print(f"[Config] voice_a={va} | voice_b={vb}")

    status = (
        f"✅ Configurações salvas!\n\n"
        f"🤖 Chat:    {chat_m}\n"
        f"🎙️ Debate:  {podcast_m}\n"
        f"📝 Auxiliar: {summary_m}\n"
        f"👨 Voz A:   {va}\n"
        f"👩 Voz B:   {vb}"
    )
    return status, gr.Dropdown(value=va), gr.Dropdown(value=vb)


def info_sistema():
    """Retorna informações do sistema para exibir na aba Configurações."""
    linhas = []

    # GPU
    if GPU_DISPONIVEL:
        nome_gpu = torch.cuda.get_device_name(0)
        # mem_get_info considera todos os processos (inclui Ollama), não só este
        vram_livre, vram_tot = torch.cuda.mem_get_info(0)
        linhas.append(f"🖥️  GPU:          {nome_gpu}")
        linhas.append(f"💾 VRAM total:   {vram_tot / 1024**3:.1f} GB")
        linhas.append(f"💾 VRAM livre:   {vram_livre / 1024**3:.1f} GB")
    else:
        linhas.append("🖥️  GPU:          não disponível (modo CPU)")

    linhas.append("")
    linhas.append(f"🧠 Embedding:    {EMBED_DEVICE.upper()}")
    linhas.append(f"🧠 Reranker:     {RERANK_DEVICE.upper()}")
    linhas.append(f"🔊 TTS device:   {TTS_DEVICE.upper()}")
    linhas.append("")
    linhas.append(f"📡 Ollama URL:   {OLLAMA_URL}")
    linhas.append(f"🗄️  ChromaDB:     {CHROMA_HOST}:{CHROMA_PORT}")
    linhas.append(f"📁 Documentos:   {DOCUMENTS_PATH}")
    linhas.append(f"🎵 Áudios:       {AUDIO_PATH}")
    linhas.append("")

    # Índice
    try:
        with abrir_indice() as con:
            arquivos, pais, filhos = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(pais), 0), COALESCE(SUM(filhos), 0) FROM arquivos"
            ).fetchone()
        linhas.append(f"📚 Indexados:    {arquivos} arquivo(s) | {pais} pais | {filhos} filhos")
    except Exception as e:
        linhas.append(f"📚 Índice:       erro ao ler ({e})")
    linhas.append("")

    # Modelos ativos
    linhas.append(f"🤖 Chat ativo:     {_config['chat_model']}")
    linhas.append(f"🎙️ Debate ativo:   {_config['podcast_model']}")
    linhas.append(f"📝 Auxiliar ativo: {_config['summary_model']}")
    linhas.append(f"👨 Voz A ativa:    {_config['voice_a']}")
    linhas.append(f"👩 Voz B ativa:    {_config['voice_b']}")

    # Modelos Ollama disponíveis
    modelos = listar_modelos_ollama()
    linhas.append("")
    linhas.append(f"📦 Modelos no Ollama ({len(modelos)}):")
    for m in modelos:
        linhas.append(f"   • {m}")

    return "\n".join(linhas)


# ============================================================
# KOKORO
# ============================================================

kokoro_pipeline = None
kokoro_lock = threading.Lock()


def inicializar_kokoro():
    global kokoro_pipeline
    if kokoro_pipeline is not None:
        return kokoro_pipeline
    with kokoro_lock:
        if kokoro_pipeline is not None:
            return kokoro_pipeline
        print(f"Inicializando Kokoro em {TTS_DEVICE}...")
        try:
            kokoro_pipeline = KPipeline(lang_code="p", device=TTS_DEVICE)
            print("Kokoro inicializado com sucesso.")
            return kokoro_pipeline
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Não foi possível inicializar o Kokoro: {e}")


# ============================================================
# CHROMA CLIENT
# ============================================================

_chroma_client = None
_colecao = None
_chroma_lock = threading.Lock()


def obter_cliente_chroma():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_client


def obter_colecao(recarregar=False):
    """Coleção dos fragmentos filhos, reutilizada entre requisições."""
    global _colecao
    with _chroma_lock:
        if _colecao is None or recarregar:
            _colecao = obter_cliente_chroma().get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )
        return _colecao


def resetar_colecao():
    """Apaga e recria a coleção de fragmentos."""
    try:
        obter_cliente_chroma().delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    return obter_colecao(recarregar=True)


def remover_colecoes_legadas():
    """Remove coleções do formato antigo (guardavam o texto pai em cada filho)."""
    removidas = []
    cliente = obter_cliente_chroma()
    for nome in LEGACY_COLLECTIONS:
        try:
            cliente.delete_collection(name=nome)
            removidas.append(nome)
        except Exception:
            pass
    return removidas


# ============================================================
# ÍNDICE LOCAL (blocos pais + controle de arquivos indexados)
# ============================================================

def abrir_indice():
    """
    Abre o SQLite com os blocos pais e o hash de cada arquivo indexado.
    Uso: with abrir_indice() as con: ...  (commit e close automáticos)
    """
    con = sqlite3.connect(INDEX_DB_PATH, timeout=30)
    con.execute(
        "CREATE TABLE IF NOT EXISTS pais ("
        "id TEXT PRIMARY KEY, source TEXT, page TEXT, texto TEXT)"
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_pais_source ON pais(source)")
    con.execute(
        "CREATE TABLE IF NOT EXISTS arquivos ("
        "source TEXT PRIMARY KEY, hash TEXT, pais INTEGER, filhos INTEGER, indexado_em TEXT)"
    )
    return _ConexaoIndice(con)


class _ConexaoIndice:
    """Context manager que faz commit/rollback e sempre fecha a conexão."""

    def __init__(self, con):
        self.con = con

    def __enter__(self):
        return self.con

    def __exit__(self, tipo, *_):
        with closing(self.con):
            if tipo is None:
                self.con.commit()
            else:
                self.con.rollback()
        return False


def hash_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def hashes_indexados():
    with abrir_indice() as con:
        return dict(con.execute("SELECT source, hash FROM arquivos"))


def remover_do_indice(nome, colecao=None, con=None):
    """Remove todos os fragmentos e blocos pais de um arquivo."""
    if con is None:
        with abrir_indice() as con_nova:
            remover_do_indice(nome, colecao, con_nova)
        return
    colecao = colecao or obter_colecao()
    colecao.delete(where={"source": nome})
    con.execute("DELETE FROM pais WHERE source = ?", (nome,))
    con.execute("DELETE FROM arquivos WHERE source = ?", (nome,))


def buscar_pais(parent_ids):
    """Retorna {id: {"texto", "source", "page"}} para os ids pedidos."""
    if not parent_ids:
        return {}
    marcadores = ",".join("?" for _ in parent_ids)
    with abrir_indice() as con:
        linhas = con.execute(
            f"SELECT id, source, page, texto FROM pais WHERE id IN ({marcadores})",
            list(parent_ids)
        ).fetchall()
    return {
        pid: {"source": source, "page": page, "texto": texto}
        for pid, source, page, texto in linhas
    }


# ============================================================
# GERENCIAMENTO DE DOCUMENTOS
# ============================================================

EXTENSOES_PERMITIDAS = [".pdf", ".txt", ".docx"]


def listar_documentos():
    """Retorna lista de nomes de arquivos válidos na pasta documentos."""
    if not os.path.exists(DOCUMENTS_PATH):
        return []
    arquivos = []
    for arquivo in Path(DOCUMENTS_PATH).iterdir():
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_PERMITIDAS:
            arquivos.append(arquivo.name)
    return sorted(arquivos)


def formatar_tamanho(tamanho):
    if tamanho < 1024:
        return f"{tamanho} B"
    if tamanho < 1024 * 1024:
        return f"{tamanho / 1024:.1f} KB"
    return f"{tamanho / (1024*1024):.1f} MB"


def atualizar_lista_documentos():
    """Retorna as linhas da tabela de documentos, com o status no índice."""
    try:
        indexados = hashes_indexados()
    except Exception:
        indexados = {}

    linhas = []
    for nome in listar_documentos():
        path = os.path.join(DOCUMENTS_PATH, nome)
        ext = Path(nome).suffix.lower().replace(".", "").upper()
        if nome not in indexados:
            status = "⏳ Pendente"
        elif indexados[nome] != hash_arquivo(path):
            status = "🔁 Alterado"
        else:
            status = "✅ Indexado"
        linhas.append([nome, formatar_tamanho(os.path.getsize(path)), ext, status])
    return linhas


def upload_documentos(arquivos):
    """
    Recebe lista de arquivos via drag-and-drop ou botão Add,
    copia para a pasta documentos e retorna status + lista atualizada.
    """
    if not arquivos:
        return "Nenhum arquivo recebido.", atualizar_lista_documentos()

    copiados = []
    erros = []

    for arquivo in arquivos:
        # Gradio entrega caminhos (str); versões antigas entregavam objetos com .name
        origem = getattr(arquivo, "name", arquivo)
        nome = os.path.basename(origem)
        try:
            ext = Path(nome).suffix.lower()

            if ext not in EXTENSOES_PERMITIDAS:
                erros.append(f"❌ {nome} — formato não suportado (use PDF, TXT ou DOCX)")
                continue

            destino = os.path.join(DOCUMENTS_PATH, nome)
            shutil.copy2(origem, destino)
            copiados.append(f"✅ {nome}")

        except Exception as e:
            erros.append(f"❌ {nome} — erro: {str(e)}")

    linhas = copiados + erros
    status = "\n".join(linhas) if linhas else "Nenhum arquivo processado."
    if copiados:
        status += "\n\nClique em 🔄 Processar e Vetorizar para indexar."

    return status, atualizar_lista_documentos()


def formatar_selecao(selecionados):
    if not selecionados:
        return "Nenhum documento selecionado."
    return "\n".join(f"• {nome}" for nome in selecionados)


def alternar_selecao(selecionados, evt: gr.SelectData):
    """Clique numa linha da tabela marca/desmarca o documento."""
    nome = None
    if evt.row_value:
        nome = evt.row_value[0]
    if not nome:
        return selecionados, formatar_selecao(selecionados)

    selecionados = list(selecionados or [])
    if nome in selecionados:
        selecionados.remove(nome)
    else:
        selecionados.append(nome)
    return selecionados, formatar_selecao(selecionados)


def limpar_selecao():
    return [], formatar_selecao([])


def remover_documentos(selecionados):
    """
    Remove os arquivos selecionados da pasta documentos
    e os respectivos fragmentos do índice.
    """
    if not selecionados:
        return (
            "⚠️ Clique nas linhas da tabela para selecionar documentos.",
            atualizar_lista_documentos(), [], formatar_selecao([])
        )

    removidos = []
    erros = []

    for nome in selecionados:
        # basename impede apagar algo fora da pasta documentos
        nome = os.path.basename(nome)
        path = os.path.join(DOCUMENTS_PATH, nome)
        try:
            if os.path.exists(path):
                os.remove(path)
            remover_do_indice(nome)
            removidos.append(f"🗑️ {nome} removido (arquivo e índice)")
        except Exception as e:
            erros.append(f"❌ {nome} — erro: {str(e)}")

    linhas = removidos + erros
    status = "\n".join(linhas) if linhas else "Nenhum arquivo removido."

    return status, atualizar_lista_documentos(), [], formatar_selecao([])


# ============================================================
# INDEXAÇÃO
# ============================================================

PARENT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=200)

# Duas granularidades: fragmentos curtos (precisão) e médios (contexto).
# Mais splitters só geravam fragmentos quase duplicados do mesmo pai.
CHILD_SPLITTERS = [
    RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80),
    RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150),
]


def carregar_paginas(caminho):
    """Retorna lista de (página, texto). Página vazia quando não se aplica."""
    ext = caminho.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(caminho))
        return [(str(i + 1), pagina.extract_text() or "") for i, pagina in enumerate(reader.pages)]
    if ext == ".txt":
        return [("", caminho.read_text(encoding="utf-8", errors="ignore"))]
    if ext == ".docx":
        return [("", docx2txt.process(str(caminho)) or "")]
    return []


def indexar_arquivo(caminho, hash_atual, colecao, con):
    """Indexa um arquivo (substituindo a versão anterior). Retorna (pais, filhos)."""
    nome = caminho.name
    remover_do_indice(nome, colecao, con)

    pais = []
    ids, textos, metadados = [], [], []

    for pagina, texto in carregar_paginas(caminho):
        for parent_text in PARENT_SPLITTER.split_text(texto):
            if len(parent_text.strip()) <= 10:
                continue
            parent_id = str(uuid.uuid4())
            pais.append((parent_id, nome, pagina, parent_text))

            for splitter in CHILD_SPLITTERS:
                for child_text in splitter.split_text(parent_text):
                    if len(child_text.strip()) <= 10:
                        continue
                    ids.append(str(uuid.uuid4()))
                    textos.append(child_text)
                    metadados.append({"parent_id": parent_id, "source": nome, "page": pagina})

    if textos:
        vetores = embed_documentos(textos)
        for i in range(0, len(ids), CHROMA_BATCH):
            colecao.add(
                ids=ids[i:i + CHROMA_BATCH],
                embeddings=vetores[i:i + CHROMA_BATCH],
                documents=textos[i:i + CHROMA_BATCH],
                metadatas=metadados[i:i + CHROMA_BATCH]
            )

    con.executemany("INSERT INTO pais (id, source, page, texto) VALUES (?, ?, ?, ?)", pais)
    con.execute(
        "INSERT OR REPLACE INTO arquivos (source, hash, pais, filhos, indexado_em) "
        "VALUES (?, ?, ?, ?, ?)",
        (nome, hash_atual, len(pais), len(ids), datetime.datetime.now().isoformat(timespec="seconds"))
    )
    return len(pais), len(ids)


def indexar_documentos(reindexar_tudo=False):
    """
    Indexação incremental: só processa arquivos novos ou alterados
    e remove do índice os arquivos que saíram da pasta.
    """
    linhas = []

    def status():
        return "\n".join(linhas), atualizar_lista_documentos()

    try:
        legadas = remover_colecoes_legadas()
        if legadas:
            linhas.append(f"🧹 Coleção antiga removida: {', '.join(legadas)}")

        if reindexar_tudo:
            colecao = resetar_colecao()
            with abrir_indice() as con:
                con.execute("DELETE FROM pais")
                con.execute("DELETE FROM arquivos")
            linhas.append("♻️ Índice apagado — reindexando tudo.")
        else:
            colecao = obter_colecao(recarregar=True)

        arquivos = listar_documentos()
        indexados = hashes_indexados()

        removidos = sorted(set(indexados) - set(arquivos))
        for nome in removidos:
            remover_do_indice(nome, colecao)
            linhas.append(f"🗑️ {nome} — removido do índice (não está mais na pasta)")

        if not arquivos:
            linhas.append("Nenhum arquivo encontrado em data/documentos.")
            yield status()
            return

        total_pais = total_filhos = inalterados = 0

        for nome in arquivos:
            caminho = Path(DOCUMENTS_PATH) / nome
            try:
                hash_atual = hash_arquivo(caminho)
                # Confere também no Chroma, caso o volume tenha sido apagado
                if (
                    indexados.get(nome) == hash_atual
                    and colecao.get(where={"source": nome}, limit=1)["ids"]
                ):
                    inalterados += 1
                    continue

                linhas.append(f"⏳ {nome} — indexando na {EMBED_DEVICE.upper()}...")
                yield status()

                with abrir_indice() as con:
                    pais, filhos = indexar_arquivo(caminho, hash_atual, colecao, con)

                total_pais += pais
                total_filhos += filhos
                linhas[-1] = f"✅ {nome} — {pais} blocos pais, {filhos} fragmentos"
                if pais == 0:
                    linhas[-1] += " (nenhum texto extraído — PDF escaneado?)"
                yield status()

            except Exception as e:
                traceback.print_exc()
                linhas.append(f"❌ {nome} — erro: {e}")
                yield status()

        liberar_vram()

        linhas.append("")
        linhas.append(
            f"Concluído: {total_pais} blocos pais e {total_filhos} fragmentos novos"
            f" | {inalterados} arquivo(s) sem alteração."
        )
        yield status()

    except Exception as e:
        traceback.print_exc()
        linhas.append(f"Erro na indexação:\n{str(e)}")
        yield status()


# ============================================================
# RAG
# ============================================================

def recuperar_contexto(pergunta, numero_pais=3):
    """
    Busca fragmentos filhos, agrupa por bloco pai, reranqueia os pais
    e retorna (contexto, fontes).
    """
    colecao = obter_colecao()
    total = colecao.count()
    if total == 0:
        return None, []

    resultado = colecao.query(
        query_embeddings=[embed_pergunta(pergunta)],
        n_results=min(max(30, numero_pais * 8), total),
        include=["metadatas"]
    )

    # Pais únicos, na ordem de relevância vetorial
    parent_ids = []
    for meta in resultado["metadatas"][0]:
        pid = meta.get("parent_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)

    parent_ids = parent_ids[:max(15, numero_pais * 3)]
    pais = buscar_pais(parent_ids)
    candidatos = [pais[pid] for pid in parent_ids if pid in pais]

    if not candidatos:
        return None, []

    notas = reranker.predict(
        [[pergunta, c["texto"]] for c in candidatos],
        batch_size=16,
        show_progress_bar=False
    )

    ranqueados = sorted(zip(notas, candidatos), key=lambda x: x[0], reverse=True)[:numero_pais]

    contexto = "\n\n--- FIM DO TRECHO / NOVO TRECHO ---\n\n".join(
        c["texto"] for _, c in ranqueados
    )

    fontes = []
    for _, c in ranqueados:
        fonte = c["source"]
        if c["page"]:
            fonte = f"{fonte} (página {c['page']})"
        if fonte and fonte not in fontes:
            fontes.append(fonte)

    return contexto, fontes


# ============================================================
# OLLAMA
# ============================================================

def ollama_chat_stream(mensagens, model, num_ctx=CHAT_NUM_CTX, temperature=None):
    """Streaming via /api/chat, com o modo 'thinking' desligado (qwen3)."""
    options = {"num_ctx": num_ctx}
    if temperature is not None:
        options["temperature"] = temperature

    payload = {
        "model":    model,
        "messages": mensagens,
        "stream":   True,
        "think":    False,
        "options":  options
    }

    with requests.post(OLLAMA_URL + "/api/chat", json=payload, stream=True, timeout=(10, 600)) as resp:
        if not resp.ok:
            print(f"[Ollama Error] {resp.status_code}: {resp.text}")
        resp.raise_for_status()

        for linha in resp.iter_lines():
            if not linha:
                continue
            dado = json.loads(linha)
            if "error" in dado:
                raise RuntimeError(dado["error"])
            pedaco = dado.get("message", {}).get("content", "")
            if pedaco:
                yield pedaco
            if dado.get("done"):
                break


def ollama_chat(mensagens, model, **kwargs):
    return "".join(ollama_chat_stream(mensagens, model, **kwargs))


def ollama_json(prompt, model, temperature=0.6, num_ctx=16384, num_predict=6000):
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "think":  False,
        "options": {
            "temperature":    temperature,
            "top_p":          0.9,
            "num_ctx":        num_ctx,
            "num_predict":    num_predict,
            "repeat_penalty": 1.05
        }
    }

    response = requests.post(OLLAMA_URL + "/api/generate", json=payload, timeout=1800)
    if not response.ok:
        print(f"[Ollama Error] {response.status_code}: {response.text}")
    response.raise_for_status()

    data = response.json()
    texto = data.get("response", "")

    if not texto:
        raise RuntimeError("O Ollama retornou resposta vazia.")

    if data.get("done_reason") == "length":
        print(f"[Ollama] Resposta cortada por num_predict={num_predict}.")

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio = texto.find("{")
        fim = texto.rfind("}")
        if inicio >= 0 and fim > inicio:
            return json.loads(texto[inicio:fim + 1])
        raise RuntimeError("O modelo não retornou JSON válido.")


# ============================================================
# CHAT
# ============================================================

def formatar_historico(historico, max_mensagens=4, max_chars=1500):
    """Converte o histórico (formato messages do Gradio) em texto curto."""
    linhas = []
    for msg in (historico or [])[-max_mensagens:]:
        if not isinstance(msg, dict):
            continue
        conteudo = msg.get("content")
        if not isinstance(conteudo, str) or not conteudo.strip():
            continue
        # Não repete o bloco de fontes das respostas anteriores
        conteudo = conteudo.split(SEPARADOR_FONTES)[0].strip()[:max_chars]
        papel = "Usuário" if msg.get("role") == "user" else "Assistente"
        linhas.append(f"{papel}: {conteudo}")
    return "\n".join(linhas)


def reescrever_pergunta(pergunta, historico_texto):
    """
    Transforma perguntas de continuação ("e o prazo disso?") em perguntas
    independentes, para que a busca vetorial encontre os trechos certos.
    """
    prompt = f"""Dada a conversa abaixo e uma nova pergunta, reescreva a nova pergunta para que ela seja compreensível sozinha, sem a conversa.
Substitua pronomes e referências vagas ("isso", "ele", "o prazo") pelos termos a que se referem.
Se a pergunta já for independente, repita-a sem alterações.
Responda APENAS com a pergunta reescrita, sem explicações.

CONVERSA:
{historico_texto}

NOVA PERGUNTA: {pergunta}

PERGUNTA REESCRITA:"""

    try:
        reescrita = ollama_chat(
            [{"role": "user", "content": prompt}],
            model=_config["summary_model"],
            num_ctx=4096,
            temperature=0
        ).strip().strip('"').strip()
    except Exception as e:
        print(f"[Reescrita] falhou, usando pergunta original: {e}")
        return pergunta

    if not reescrita or len(reescrita) > 500:
        return pergunta
    return reescrita


PROMPT_SISTEMA_CHAT = """Você é um auditor especialista em leitura de editais e manuais técnicos.

Responda à pergunta do usuário baseando-se ESTRITAMENTE no contexto fornecido.

Se a resposta não estiver clara no contexto, diga explicitamente que não possui essa informação.

Não invente. Não presuma."""


def responder_pergunta(pergunta, historico):
    try:
        historico_texto = formatar_historico(historico)
        pergunta_busca = pergunta

        if historico_texto:
            yield "⏳ Interpretando a pergunta com base na conversa..."
            pergunta_busca = reescrever_pergunta(pergunta, historico_texto)
            if pergunta_busca != pergunta:
                print(f"[Reescrita] {pergunta!r} -> {pergunta_busca!r}")

        yield "🔍 Buscando e reranqueando fragmentos no ChromaDB..."

        contexto, fontes = recuperar_contexto(pergunta_busca, numero_pais=3)

        if not contexto:
            yield "Nenhum documento relevante foi encontrado. Os documentos já foram vetorizados?"
            return

        yield f"⚡ {len(fontes)} fonte(s) relevante(s). Enviando ao modelo {_config['chat_model']}..."

        mensagem_usuario = f"""[HISTÓRICO RECENTE DA CONVERSA]

{historico_texto if historico_texto else "Início da conversa."}

[CONTEXTO RECUPERADO]

{contexto}

[PERGUNTA]

{pergunta}"""

        mensagens = [
            {"role": "system", "content": PROMPT_SISTEMA_CHAT},
            {"role": "user", "content": mensagem_usuario},
        ]

        resposta_parcial = ""
        for chunk in ollama_chat_stream(mensagens, model=_config["chat_model"]):
            resposta_parcial += chunk
            yield resposta_parcial

        yield (
            resposta_parcial
            + SEPARADOR_FONTES + "\n"
            + "\n".join(f"- {f}" for f in fontes)
        )

    except Exception as e:
        traceback.print_exc()
        yield f"Erro no processamento: {str(e)}"


# ============================================================
# TTS HELPERS
# ============================================================

def limpar_texto_tts(texto):
    if not texto:
        return ""
    texto = str(texto)
    texto = re.sub(r"\*\*(.*?)\*\*", r"\1", texto)
    texto = re.sub(r"\*(.*?)\*",     r"\1", texto)
    texto = re.sub(r"`(.*?)`",       r"\1", texto)
    texto = texto.replace("\n", " ")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def gerar_fala(texto, voice, velocidade=1.0):
    pipeline = inicializar_kokoro()
    texto = limpar_texto_tts(texto)
    if not texto:
        return np.zeros(2400, dtype=np.float32)
    partes = []
    with kokoro_lock:
        generator = pipeline(texto, voice=voice, speed=velocidade, split_pattern=r"\n+")
        for _, _, audio in generator:
            if audio is not None:
                partes.append(np.asarray(audio, dtype=np.float32))
    if not partes:
        raise RuntimeError(f"Kokoro não gerou áudio para a voz {voice}.")
    return np.concatenate(partes)


def silencio(segundos, sample_rate=SAMPLE_RATE):
    return np.zeros(int(segundos * sample_rate), dtype=np.float32)


def montar_audio(dialogo, voice_a, voice_b, velocidade=1.0):
    audio_partes = []
    for item in dialogo:
        speaker = str(item.get("speaker", "")).lower()
        texto   = item.get("text", "")
        if not texto:
            continue
        voice = voice_a if speaker in ["professor", "a", "speaker_a"] else voice_b
        audio_partes.append(gerar_fala(texto, voice, velocidade))
        audio_partes.append(silencio(0.20))
    if not audio_partes:
        raise RuntimeError("Nenhuma fala foi gerada.")
    return np.concatenate(audio_partes)


def salvar_mp3(audio, mp3_path, sample_rate=SAMPLE_RATE):
    """Envia o áudio direto ao FFmpeg (sem gravar WAV intermediário)."""
    resultado = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-codec:a", "libmp3lame", "-b:a", "128k", mp3_path
        ],
        input=np.asarray(audio, dtype=np.float32).tobytes(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if resultado.returncode != 0:
        raise RuntimeError("Erro no FFmpeg:\n" + resultado.stderr.decode(errors="ignore"))
    return mp3_path


# ============================================================
# GERADOR DE DEBATE
# ============================================================

def gerar_debate(tema, duracao, estilo, voice_a, voice_b, velocidade, fazer_pegadinhas, fazer_exemplos):
    try:
        if not tema or not tema.strip():
            yield None, None, "❌ Informe o tema.", ""
            return

        modelo = _config["podcast_model"]

        yield None, None, "🔍 Consultando o ChromaDB e reranqueando trechos...", ""

        duracao_int = int(duracao)
        metas_palavras = {5: 700, 10: 1400, 15: 2100, 20: 2800}
        palavras = metas_palavras.get(duracao_int, 1400)

        contexto, fontes = recuperar_contexto(tema, numero_pais=10)

        if not contexto:
            yield None, None, "❌ Nenhum documento relevante encontrado.", ""
            return

        lista_fontes = "\n".join(f"• {x}" for x in fontes)

        pegadinhas = (
            "Inclua possíveis pegadinhas de concurso quando existirem no contexto."
            if fazer_pegadinhas else "Não force pegadinhas artificiais."
        )
        exemplos = (
            "Inclua exemplos práticos para facilitar a compreensão."
            if fazer_exemplos else "Use exemplos apenas quando forem realmente úteis."
        )

        prompt = f"""
Você é um roteirista especializado em preparação para concursos públicos brasileiros.
Crie um episódio de estudo baseado exclusivamente nos documentos recuperados pelo sistema RAG.

TEMA: {tema}

PERSONAGEM A: Professor especialista — técnico, didático, paciente, objetivo.
PERSONAGEM B: Aluna inteligente e questionadora — faz perguntas, demonstra dúvidas.

ESTILO: {estilo}

REGRAS:
1. Use somente informações presentes no CONTEXTO.
2. Não invente leis, artigos, números ou exemplos.
3. O diálogo deve ensinar. Evite palestra.
4. A conversa deve parecer natural.
5. Alterne entre professor e aluna.
6. Evite repetição.
7. {pegadinhas}
8. {exemplos}

TAMANHO: Aproximadamente {palavras} palavras.

FORMATO — retorne EXCLUSIVAMENTE JSON válido:
{{
  "titulo": "Título do episódio",
  "dialogo": [
    {{"speaker": "professor", "text": "Fala do professor"}},
    {{"speaker": "aluna",     "text": "Fala da aluna"}}
  ]
}}

Não coloque markdown. Não coloque ```json. Não coloque comentários.

CONTEXTO DOS DOCUMENTOS:
{contexto}
"""

        yield (
            None, None,
            f"🧠 RAG selecionou {len(fontes)} fonte(s). Gerando roteiro com {modelo}...",
            lista_fontes
        )

        # Português gasta ~2 tokens por palavra, mais a estrutura do JSON
        resultado = ollama_json(
            prompt=prompt, model=modelo, temperature=0.65,
            num_ctx=32768, num_predict=palavras * 3 + 1000
        )

        titulo  = resultado.get("titulo",  "Debate de Estudo")
        dialogo = resultado.get("dialogo", [])

        if not dialogo:
            raise RuntimeError("O modelo não retornou diálogo.")

        dialogo_limpo = []
        for item in dialogo:
            if not isinstance(item, dict):
                continue
            speaker = item.get("speaker", "")
            text    = item.get("text",    "")
            if not text:
                continue
            if speaker not in ["professor", "aluna", "a", "b", "speaker_a", "speaker_b"]:
                continue
            dialogo_limpo.append({"speaker": speaker, "text": limpar_texto_tts(text)})

        if not dialogo_limpo:
            raise RuntimeError("O diálogo retornado não possui falas válidas.")

        yield (
            None, None,
            f"🎙️ Roteiro pronto: {len(dialogo_limpo)} falas.\n🔊 Gerando vozes com Kokoro em {TTS_DEVICE}...",
            lista_fontes
        )

        audio_array = montar_audio(
            dialogo_limpo, voice_a=voice_a, voice_b=voice_b, velocidade=float(velocidade)
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug      = re.sub(r"[^a-zA-Z0-9_-]+", "_", titulo)[:60]
        base      = f"{timestamp}_{slug}"
        mp3_path  = os.path.join(AUDIO_PATH, f"{base}.mp3")
        json_path = os.path.join(AUDIO_PATH, f"{base}.json")

        yield None, None, "🎧 Convertendo áudio para MP3...", lista_fontes

        salvar_mp3(audio_array, mp3_path)

        roteiro_completo = {
            "titulo": titulo, "tema": tema, "modelo": modelo,
            "voz_professor": voice_a, "voz_aluna": voice_b,
            "dialogo": dialogo_limpo, "fontes": fontes
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(roteiro_completo, f, ensure_ascii=False, indent=2)

        duracao_real = len(audio_array) / SAMPLE_RATE
        minutos      = int(duracao_real // 60)
        segundos_val = int(duracao_real % 60)

        yield (
            mp3_path, json_path,
            f"✅ Debate concluído!\n\n🎙️ {titulo}\n⏱️ {minutos:02d}:{segundos_val:02d}\n\n"
            f"🧠 Modelo: {modelo}\n🔊 TTS: Kokoro\n👨 {voice_a}\n👩 {voice_b}\n",
            lista_fontes
        )

    except Exception as e:
        traceback.print_exc()
        yield None, None, f"❌ ERRO AO GERAR DEBATE\n\n{str(e)}", ""


# ============================================================
# TESTE DE VOZ
# ============================================================

def testar_voz(voice):
    try:
        texto = (
            "Olá. Este é um teste de voz do seu sistema local de estudos. "
            "Estamos utilizando o Kokoro para gerar áudio em português brasileiro."
        )
        audio = gerar_fala(texto, voice)
        # Retorna o áudio em memória — não acumula arquivos de teste em disco
        return (SAMPLE_RATE, audio), f"✅ Voz {voice} funcionando."
    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Erro: {str(e)}"


# ============================================================
# INTERFACE GRADIO
# ============================================================

with gr.Blocks(title="Advanced RAG Corporativo") as demo:

    gr.Markdown("""
# 🚀 Advanced RAG Corporativo
### RAG local + Ollama + ChromaDB + Podcast de Estudo
Sistema totalmente local para consulta de documentos e geração de debates em áudio.
""")

    with gr.Tabs():

        # ====================================================
        # ABA DOCUMENTOS
        # ====================================================

        with gr.Tab("📁 Documentos"):

            gr.Markdown("""
## 📁 Gerenciamento de Documentos

Arraste arquivos para a área abaixo ou clique em **Adicionar** para enviar do seu computador.
Após enviar, clique em **🔄 Processar e Vetorizar** — só arquivos novos ou alterados são processados.

Formatos aceitos: **PDF, TXT, DOCX**
""")

            with gr.Row():

                # ── Coluna esquerda: upload ──────────────────
                with gr.Column(scale=1):

                    gr.Markdown("### ⬆️ Enviar documentos")

                    upload_area = gr.File(
                        label="Arraste arquivos aqui ou clique em Adicionar",
                        file_count="multiple",
                        file_types=EXTENSOES_PERMITIDAS,
                        height=200,
                    )

                    btn_upload = gr.Button(
                        "⬆️ Enviar para pasta documentos",
                        variant="secondary"
                    )

                    status_upload = gr.Textbox(
                        label="Status do envio",
                        lines=5,
                        interactive=False
                    )

                    gr.Markdown("---")

                    gr.Markdown("### ⚙️ Vetorizar")

                    reindexar_tudo = gr.Checkbox(
                        value=False,
                        label="♻️ Reindexar tudo do zero",
                        info="Use após mudar o modelo de embedding ou a estratégia de chunking"
                    )

                    btn_indexar = gr.Button(
                        "🔄 Processar e Vetorizar no ChromaDB",
                        variant="primary",
                        size="lg"
                    )

                    status_indexacao = gr.Textbox(
                        label="Status da vetorização",
                        lines=8,
                        interactive=False
                    )

                # ── Coluna direita: lista de documentos ──────
                with gr.Column(scale=1):

                    gr.Markdown("### 📋 Documentos na pasta")

                    tabela_docs = gr.Dataframe(
                        headers=["Arquivo", "Tamanho", "Tipo", "Índice"],
                        datatype=["str", "str", "str", "str"],
                        interactive=False,
                        label="Clique nas linhas para selecionar/desmarcar",
                        max_height=600,
                        col_count=(4, "fixed"),
                    )

                    selecionados = gr.State([])

                    txt_selecionados = gr.Textbox(
                        label="Selecionados para remoção",
                        value=formatar_selecao([]),
                        lines=3,
                        interactive=False
                    )

                    with gr.Row():

                        btn_atualizar = gr.Button(
                            "🔃 Atualizar lista",
                            variant="secondary",
                            size="sm"
                        )

                        btn_limpar_selecao = gr.Button(
                            "✖️ Limpar seleção",
                            variant="secondary",
                            size="sm"
                        )

                        btn_deletar = gr.Button(
                            "🗑️ Remover selecionados",
                            variant="stop",
                            size="sm"
                        )

                    status_remocao = gr.Textbox(
                        label="Status da remoção",
                        lines=3,
                        interactive=False
                    )

            # ── Eventos ──────────────────────────────────────

            btn_upload.click(
                fn=upload_documentos,
                inputs=[upload_area],
                outputs=[status_upload, tabela_docs]
            )

            btn_indexar.click(
                fn=indexar_documentos,
                inputs=[reindexar_tudo],
                outputs=[status_indexacao, tabela_docs]
            )

            btn_atualizar.click(
                fn=atualizar_lista_documentos,
                outputs=[tabela_docs]
            ).then(
                fn=limpar_selecao,
                outputs=[selecionados, txt_selecionados]
            )

            tabela_docs.select(
                fn=alternar_selecao,
                inputs=[selecionados],
                outputs=[selecionados, txt_selecionados]
            )

            btn_limpar_selecao.click(
                fn=limpar_selecao,
                outputs=[selecionados, txt_selecionados]
            )

            btn_deletar.click(
                fn=remover_documentos,
                inputs=[selecionados],
                outputs=[status_remocao, tabela_docs, selecionados, txt_selecionados]
            )

            # Atualiza lista ao carregar a aba
            demo.load(
                fn=atualizar_lista_documentos,
                outputs=[tabela_docs]
            )

        # ====================================================
        # ABA CHAT
        # ====================================================

        with gr.Tab("💬 Chat RAG"):

            gr.Markdown("Faça perguntas sobre os documentos indexados no ChromaDB.")

            chatbot = gr.ChatInterface(
                fn=responder_pergunta,
                type="messages",
                title="Chatbot com RAG",
                chatbot=gr.Chatbot(type="messages", height=700),
                stop_btn="⏹️ Interromper Geração"
            )

        # ====================================================
        # ABA DEBATE
        # ====================================================

        with gr.Tab("🎙️ Debate de Estudo"):

            gr.Markdown("""
## 🎙️ Gerar Debate de Estudo

O sistema consulta o ChromaDB, recupera os trechos mais relevantes,
utiliza o modelo configurado para criar o diálogo e depois gera duas vozes
em português brasileiro com o Kokoro.
""")

            with gr.Row():

                with gr.Column(scale=1):

                    tema = gr.Textbox(
                        label="📚 Tema",
                        placeholder="Ex.: Não cumulatividade do ICMS",
                        lines=3
                    )

                    duracao = gr.Dropdown(
                        choices=[5, 10, 15, 20], value=10,
                        label="⏱️ Duração aproximada"
                    )

                    estilo = gr.Dropdown(
                        choices=[
                            "Debate didático",
                            "Professor x aluna",
                            "Debate com perguntas",
                            "Revisão para concurso",
                            "Professor x aluno cético"
                        ],
                        value="Debate didático",
                        label="🎭 Estilo"
                    )

                    voice_a = gr.Dropdown(
                        choices=OPCOES_VOZES,
                        value=_config["voice_a"],
                        label="👨 Voz do professor"
                    )

                    voice_b = gr.Dropdown(
                        choices=OPCOES_VOZES,
                        value=_config["voice_b"],
                        label="👩 Voz da aluna"
                    )

                    velocidade = gr.Slider(
                        minimum=0.85, maximum=1.15, value=1.0,
                        step=0.05, label="🎚️ Velocidade da fala"
                    )

                    fazer_pegadinhas = gr.Checkbox(
                        value=True, label="🪤 Explorar pegadinhas de concurso"
                    )

                    fazer_exemplos = gr.Checkbox(
                        value=True, label="💡 Utilizar exemplos"
                    )

                    gerar_btn = gr.Button(
                        "🎙️ GERAR DEBATE", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):

                    audio_output = gr.Audio(
                        label="🎧 Podcast", format="mp3",
                        autoplay=False, show_download_button=True
                    )

                    roteiro_download = gr.File(label="📄 Baixar roteiro JSON")

                    debate_status = gr.Textbox(
                        label="⚙️ Status do processamento", lines=8
                    )

                    fontes_output = gr.Textbox(
                        label="📚 Fontes utilizadas", lines=8
                    )

            gerar_btn.click(
                fn=gerar_debate,
                inputs=[
                    tema, duracao, estilo,
                    voice_a, voice_b, velocidade,
                    fazer_pegadinhas, fazer_exemplos
                ],
                outputs=[audio_output, roteiro_download, debate_status, fontes_output]
            )

        # ====================================================
        # ABA TESTE DE VOZ
        # ====================================================

        with gr.Tab("🔊 Testar Vozes"):

            gr.Markdown("Use esta aba para testar o Kokoro antes de gerar um episódio completo.")

            voice_test  = gr.Dropdown(
                choices=OPCOES_VOZES,
                value="pm_alex", label="Voz"
            )
            test_btn    = gr.Button("🔊 Gerar Teste")
            test_audio  = gr.Audio(label="Resultado", format="wav")
            test_status = gr.Textbox(label="Status")

            test_btn.click(
                fn=testar_voz,
                inputs=voice_test,
                outputs=[test_audio, test_status]
            )

        # ====================================================
        # ABA CONFIGURAÇÕES
        # ====================================================

        with gr.Tab("⚙️ Configurações"):

            gr.Markdown("""
## ⚙️ Configurações do sistema

Selecione os modelos a partir dos que estão **baixados no Ollama**.
As alterações têm efeito imediato — sem necessidade de reiniciar o container.
""")

            with gr.Row():

                # ── Coluna esquerda: seleção de modelos ──────
                with gr.Column(scale=1):

                    gr.Markdown("### 🤖 Modelos de linguagem (Ollama)")

                    cfg_chat_model = gr.Dropdown(
                        choices=[_config["chat_model"]],
                        value=_config["chat_model"],
                        label="🤖 Modelo do Chat RAG",
                        info="Usado nas perguntas da aba Chat"
                    )

                    cfg_podcast_model = gr.Dropdown(
                        choices=[_config["podcast_model"]],
                        value=_config["podcast_model"],
                        label="🎙️ Modelo do Debate / Podcast",
                        info="Usado para gerar o roteiro do debate"
                    )

                    cfg_summary_model = gr.Dropdown(
                        choices=[_config["summary_model"]],
                        value=_config["summary_model"],
                        label="📝 Modelo auxiliar",
                        info="Modelo leve que reescreve perguntas de continuação do chat"
                    )

                    gr.Markdown("### 🔊 Vozes Kokoro (TTS)")

                    cfg_voice_a = gr.Dropdown(
                        choices=OPCOES_VOZES,
                        value=_config["voice_a"],
                        label="👨 Voz do Professor",
                        info="Voz padrão do personagem A na aba Debate"
                    )

                    cfg_voice_b = gr.Dropdown(
                        choices=OPCOES_VOZES,
                        value=_config["voice_b"],
                        label="👩 Voz da Aluna",
                        info="Voz padrão do personagem B na aba Debate"
                    )

                    with gr.Row():

                        btn_recarregar_modelos = gr.Button(
                            "🔃 Recarregar modelos do Ollama",
                            variant="secondary",
                            size="sm"
                        )

                        btn_salvar_config = gr.Button(
                            "💾 Salvar configurações",
                            variant="primary",
                            size="sm"
                        )

                    status_config = gr.Textbox(
                        label="Status",
                        lines=8,
                        interactive=False
                    )

                # ── Coluna direita: info do sistema ──────────
                with gr.Column(scale=1):

                    gr.Markdown("### 🖥️ Informações do sistema")

                    info_box = gr.Textbox(
                        label="Status atual",
                        lines=25,
                        interactive=False,
                        value=info_sistema
                    )

                    btn_refresh_info = gr.Button(
                        "🔃 Atualizar informações",
                        variant="secondary",
                        size="sm"
                    )

            # ── Eventos ──────────────────────────────────────

            btn_recarregar_modelos.click(
                fn=recarregar_e_atualizar_dropdowns,
                outputs=[
                    cfg_chat_model,
                    cfg_podcast_model,
                    cfg_summary_model,
                    status_config
                ]
            )

            btn_salvar_config.click(
                fn=salvar_configuracoes,
                inputs=[
                    cfg_chat_model,
                    cfg_podcast_model,
                    cfg_summary_model,
                    cfg_voice_a,
                    cfg_voice_b
                ],
                outputs=[status_config, voice_a, voice_b]
            )

            btn_refresh_info.click(
                fn=info_sistema,
                outputs=[info_box]
            )

            # Carrega modelos do Ollama ao abrir a aba
            demo.load(
                fn=recarregar_e_atualizar_dropdowns,
                outputs=[
                    cfg_chat_model,
                    cfg_podcast_model,
                    cfg_summary_model,
                    status_config
                ]
            )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("INICIANDO ADVANCED RAG")
    print("=" * 70)
    print(f"GPU disponível:  {GPU_DISPONIVEL}")

    if GPU_DISPONIVEL:
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
        print(f"VRAM total:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print(f"Embedding:       {EMBED_DEVICE.upper()}")
    print(f"Reranker:        {RERANK_DEVICE.upper()}")
    print(f"TTS device:      {TTS_DEVICE}")
    print(f"Autenticação:    usuário={AUTH_USER}")
    print("=" * 70)

    demo.queue(max_size=2)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        prevent_thread_lock=False,
        auth=(AUTH_USER, AUTH_PASS),          # ← senha ativada
        auth_message="🔐 Advanced RAG — acesso restrito. Insira suas credenciais."
    )
