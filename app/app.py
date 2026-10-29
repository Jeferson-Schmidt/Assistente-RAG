import os
import sys
import uuid
import json
import re
import traceback
import subprocess
import threading
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import requests
import torch

import gradio as gr
import chromadb

from chromadb.config import Settings

from langchain_community.llms import Ollama

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    DirectoryLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

from kokoro import KPipeline

import gradio_client.utils

# Salva a função original
_orig_json_schema_to_python_type = gradio_client.utils._json_schema_to_python_type

# Cria a função corrigida
def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)

# Substitui a função no módulo
gradio_client.utils._json_schema_to_python_type = _patched_json_schema_to_python_type


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_IMPL"] = (
    "chromadb.telemetry.dummy.DummyTelemetry"
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3:14b")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "llama3.2:1b")
PODCAST_MODEL = os.getenv("PODCAST_MODEL", "qwen3:14b")
KOKORO_DEVICE = os.getenv("KOKORO_DEVICE", "cuda")
VOICE_A = os.getenv("KOKORO_VOICE_A", "pm_alex")
VOICE_B = os.getenv("KOKORO_VOICE_B", "pf_dora")

DOCUMENTS_PATH = "/app/data/documentos"
AUDIO_PATH = "/app/audio/podcasts"
COLLECTION_NAME = "edital_avancado"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

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

# ============================================================
# CONTROLE DE GPU
# ============================================================

GPU_DISPONIVEL = torch.cuda.is_available()

TTS_DEVICE = "cuda" if (KOKORO_DEVICE == "cuda" and GPU_DISPONIVEL) else "cpu"
EMBED_INDEXACAO_DEVICE = "cuda" if GPU_DISPONIVEL else "cpu"
EMBED_RETRIEVAL_DEVICE = "cpu"

print("=" * 70)
print("SISTEMA RAG + PODCAST")
print("=" * 70)
print(f"Ollama:              {OLLAMA_URL}")
print(f"Chat model:          {CHAT_MODEL}")
print(f"Podcast LLM:         {PODCAST_MODEL}")
print(f"Embedding model:     {EMBEDDING_MODEL}")
print(f"Reranker model:      {RERANKER_MODEL}")
print(f"Embed indexação:     {EMBED_INDEXACAO_DEVICE.upper()}")
print(f"Embed retrieval:     {EMBED_RETRIEVAL_DEVICE.upper()}")
print(f"TTS device:          {TTS_DEVICE}")
print(f"Voice A:             {VOICE_A}")
print(f"Voice B:             {VOICE_B}")
print("=" * 70)

# ============================================================
# MODELOS DE EMBEDDING
# ============================================================

print("Carregando embedding para retrieval (CPU)...")

embeddings_retrieval = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": EMBED_RETRIEVAL_DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)

print("Carregando embedding para indexação (GPU se disponível)...")

if EMBED_INDEXACAO_DEVICE == "cuda":
    embeddings_indexacao = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
else:
    embeddings_indexacao = embeddings_retrieval

print("Embeddings prontos.")

# ============================================================
# RERANKER
# ============================================================

print("Carregando reranker (CPU)...")

reranker = CrossEncoder(RERANKER_MODEL, device="cpu")

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


def listar_modelos_ollama():
    """
    Consulta a API do Ollama e retorna lista de modelos baixados.
    Retorna lista de strings com os nomes dos modelos.
    """
    try:
        resp = requests.get(
            OLLAMA_URL.rstrip("/") + "/api/tags",
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        modelos = [m["name"] for m in data.get("models", [])]
        return sorted(modelos) if modelos else ["(nenhum modelo encontrado)"]
    except Exception as e:
        print(f"Erro ao listar modelos Ollama: {e}")
        return ["(erro ao conectar ao Ollama)"]


def atualizar_modelos_disponiveis():
    """
    Retorna os modelos disponíveis e os valores atuais de configuração
    para popular os dropdowns da aba Configurações.
    """
    modelos = listar_modelos_ollama()
    chat    = _config["chat_model"]
    podcast = _config["podcast_model"]
    summary = _config["summary_model"]
    va      = _config["voice_a"]
    vb      = _config["voice_b"]

    # Se o modelo configurado não estiver na lista, adiciona para não quebrar
    for m in [chat, podcast, summary]:
        if m not in modelos and not modelos[0].startswith("("):
            modelos.append(m)

    return (
        gr.Dropdown(choices=modelos, value=chat,    label="🤖 Modelo do Chat"),
        gr.Dropdown(choices=modelos, value=podcast, label="🎙️ Modelo do Debate/Podcast"),
        gr.Dropdown(choices=modelos, value=summary, label="📝 Modelo de Resumo do Histórico"),
        gr.Dropdown(choices=list(VOZES_KOKORO.keys()), value=va,
                    label="👨 Voz do Professor (Kokoro)"),
        gr.Dropdown(choices=list(VOZES_KOKORO.keys()), value=vb,
                    label="👩 Voz da Aluna (Kokoro)"),
    )


def salvar_configuracoes(chat_m, podcast_m, summary_m, va, vb):
    """
    Aplica as configurações selecionadas ao estado global.
    Não reinicia o container — efeito imediato na próxima chamada.
    """
    global kokoro_pipeline

    _config["chat_model"]    = chat_m
    _config["podcast_model"] = podcast_m
    _config["summary_model"] = summary_m
    _config["voice_a"]       = va
    _config["voice_b"]       = vb

    # Se a voz mudou, força reinicialização do Kokoro na próxima fala
    # (o pipeline do Kokoro já suporta vozes diferentes por chamada,
    #  então não precisa reiniciar — só loga a mudança)
    print(f"[Config] chat={chat_m} | podcast={podcast_m} | summary={summary_m}")
    print(f"[Config] voice_a={va} | voice_b={vb}")

    return (
        f"✅ Configurações salvas!\n\n"
        f"🤖 Chat:    {chat_m}\n"
        f"🎙️ Debate:  {podcast_m}\n"
        f"📝 Resumo:  {summary_m}\n"
        f"👨 Voz A:   {va}\n"
        f"👩 Voz B:   {vb}"
    )


def info_sistema():
    """Retorna informações do sistema para exibir na aba Configurações."""
    linhas = []

    # GPU
    if GPU_DISPONIVEL:
        nome_gpu  = torch.cuda.get_device_name(0)
        vram_tot  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_livre= (torch.cuda.get_device_properties(0).total_memory
                     - torch.cuda.memory_allocated(0)) / 1024**3
        linhas.append(f"🖥️  GPU:          {nome_gpu}")
        linhas.append(f"💾 VRAM total:   {vram_tot:.1f} GB")
        linhas.append(f"💾 VRAM livre:   {vram_livre:.1f} GB")
    else:
        linhas.append("🖥️  GPU:          não disponível (modo CPU)")

    linhas.append("")
    linhas.append(f"🧠 Embed device: {EMBED_INDEXACAO_DEVICE.upper()} (indexação)")
    linhas.append(f"🧠 Embed device: {EMBED_RETRIEVAL_DEVICE.upper()} (retrieval)")
    linhas.append(f"🔊 TTS device:   {TTS_DEVICE.upper()}")
    linhas.append("")
    linhas.append(f"📡 Ollama URL:   {OLLAMA_URL}")
    linhas.append(f"🗄️  ChromaDB:     {CHROMA_HOST}:{CHROMA_PORT}")
    linhas.append(f"📁 Documentos:   {DOCUMENTS_PATH}")
    linhas.append(f"🎵 Áudios:       {AUDIO_PATH}")
    linhas.append("")

    # Modelos ativos
    linhas.append(f"🤖 Chat ativo:    {_config['chat_model']}")
    linhas.append(f"🎙️ Debate ativo:  {_config['podcast_model']}")
    linhas.append(f"📝 Resumo ativo:  {_config['summary_model']}")
    linhas.append(f"👨 Voz A ativa:   {_config['voice_a']}")
    linhas.append(f"👩 Voz B ativa:   {_config['voice_b']}")

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

def criar_cliente_chroma():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False)
    )


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


def atualizar_lista_documentos():
    """Retorna a lista formatada para o componente Gradio."""
    docs = listar_documentos()
    if not docs:
        return gr.Dataframe(
            value=[],
            headers=["Arquivo", "Tamanho", "Tipo"],
            datatype=["str", "str", "str"]
        )
    linhas = []
    for nome in docs:
        path = os.path.join(DOCUMENTS_PATH, nome)
        tamanho = os.path.getsize(path)
        if tamanho < 1024:
            tam_str = f"{tamanho} B"
        elif tamanho < 1024 * 1024:
            tam_str = f"{tamanho / 1024:.1f} KB"
        else:
            tam_str = f"{tamanho / (1024*1024):.1f} MB"
        ext = Path(nome).suffix.lower().replace(".", "").upper()
        linhas.append([nome, tam_str, ext])
    return gr.Dataframe(
        value=linhas,
        headers=["Arquivo", "Tamanho", "Tipo"],
        datatype=["str", "str", "str"]
    )


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
        try:
            nome = os.path.basename(arquivo.name)
            ext = Path(nome).suffix.lower()

            if ext not in EXTENSOES_PERMITIDAS:
                erros.append(f"❌ {nome} — formato não suportado (use PDF, TXT ou DOCX)")
                continue

            destino = os.path.join(DOCUMENTS_PATH, nome)
            shutil.copy2(arquivo.name, destino)
            copiados.append(f"✅ {nome}")

        except Exception as e:
            erros.append(f"❌ {nome} — erro: {str(e)}")

    linhas = copiados + erros
    status = "\n".join(linhas) if linhas else "Nenhum arquivo processado."

    return status, atualizar_lista_documentos()


def remover_documentos(tabela_selecionada, selecao):
    """
    Remove arquivos selecionados na tabela da pasta documentos.
    selecao: índices das linhas selecionadas no Dataframe.
    """
    if tabela_selecionada is None or len(tabela_selecionada) == 0:
        return "Nenhum documento na lista.", atualizar_lista_documentos()

    if selecao is None or len(selecao) == 0:
        return "⚠️ Selecione pelo menos um documento na tabela.", atualizar_lista_documentos()

    removidos = []
    erros = []

    try:
        # selecao pode ser índices ou dataframe de linhas selecionadas
        indices = []
        if hasattr(selecao, 'index'):
            indices = selecao.index.tolist()
        elif isinstance(selecao, list):
            indices = selecao

        docs = listar_documentos()

        for idx in indices:
            if isinstance(idx, (int, float)) and int(idx) < len(docs):
                nome = docs[int(idx)]
                path = os.path.join(DOCUMENTS_PATH, nome)
                try:
                    os.remove(path)
                    removidos.append(f"🗑️ {nome} removido")
                except Exception as e:
                    erros.append(f"❌ {nome} — erro: {str(e)}")

    except Exception as e:
        return f"Erro ao remover: {str(e)}", atualizar_lista_documentos()

    linhas = removidos + erros
    status = "\n".join(linhas) if linhas else "Nenhum arquivo removido."

    return status, atualizar_lista_documentos()


# ============================================================
# INDEXAÇÃO
# ============================================================

def indexar_documentos():
    try:
        if not os.path.exists(DOCUMENTS_PATH) or not os.listdir(DOCUMENTS_PATH):
            return "Nenhum arquivo encontrado em data/documentos."

        pdf_loader = PyPDFDirectoryLoader(DOCUMENTS_PATH)
        txt_loader = DirectoryLoader(DOCUMENTS_PATH, glob="*.txt", loader_cls=TextLoader)
        docx_loader = DirectoryLoader(DOCUMENTS_PATH, glob="*.docx", loader_cls=Docx2txtLoader)

        docs = []
        docs.extend(pdf_loader.load())
        docs.extend(txt_loader.load())
        docs.extend(docx_loader.load())

        if not docs:
            return "Nenhum documento válido encontrado."

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, chunk_overlap=200
        )

        child_splitters = [
            RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"], chunk_size=300, chunk_overlap=50
            ),
            RecursiveCharacterTextSplitter(
                separators=[". ", "? ", "! ", "; "], chunk_size=300, chunk_overlap=50
            ),
            RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40),
            RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80),
            RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160),
        ]

        all_child_docs = []
        parents = parent_splitter.split_documents(docs)

        for parent in parents:
            parent_id = str(uuid.uuid4())
            parent_text = parent.page_content
            source_name = os.path.basename(parent.metadata.get("source", ""))
            page_number = parent.metadata.get("page", "")

            for splitter in child_splitters:
                for child_text in splitter.split_text(parent_text):
                    if len(child_text.strip()) <= 10:
                        continue
                    all_child_docs.append(Document(
                        page_content=child_text,
                        metadata={
                            "parent_id":   parent_id,
                            "parent_text": parent_text,
                            "source":      source_name,
                            "page":        str(page_number)
                        }
                    ))

        device_usado = EMBED_INDEXACAO_DEVICE.upper()
        print(f"Indexando {len(all_child_docs)} fragmentos na {device_usado}...")

        client = criar_cliente_chroma()

        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

        Chroma.from_documents(
            documents=all_child_docs,
            embedding=embeddings_indexacao,
            client=client,
            collection_name=COLLECTION_NAME
        )

        return (
            f"✅ Sucesso! ({device_usado})\n\n"
            f"Documentos carregados: {len(docs)}\n\n"
            f"Blocos pais: {len(parents)}\n\n"
            f"Fragmentos filhos: {len(all_child_docs)}"
        )

    except Exception as e:
        traceback.print_exc()
        return f"Erro na indexação:\n{str(e)}"


# ============================================================
# RAG
# ============================================================

def recuperar_contexto(pergunta, numero_pais=3, filtro_documento=None):

    client = criar_cliente_chroma()

    vectordb = Chroma(
        client=client,
        embedding_function=embeddings_retrieval,
        collection_name=COLLECTION_NAME
    )

    if filtro_documento:
        try:
            docs_filhos = vectordb.similarity_search(
                pergunta, k=20, filter={"source": filtro_documento}
            )
        except Exception:
            docs_filhos = vectordb.similarity_search(pergunta, k=20)
    else:
        docs_filhos = vectordb.similarity_search(pergunta, k=20)

    unique_parents = {}
    for doc in docs_filhos:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in unique_parents:
            unique_parents[pid] = {
                "text":   doc.metadata.get("parent_text", doc.page_content),
                "source": doc.metadata.get("source", ""),
                "page":   doc.metadata.get("page", "")
            }

    if not unique_parents:
        return None, []

    candidatos = [item["text"] for item in unique_parents.values()]
    notas = reranker.predict([[pergunta, t] for t in candidatos])

    pais_ranqueados = sorted(zip(notas, candidatos), key=lambda x: x[0], reverse=True)

    top_textos = [t for _, t in pais_ranqueados[:numero_pais]]

    contexto = "\n\n--- FIM DO TRECHO / NOVO TRECHO ---\n\n".join(top_textos)

    fontes = []
    for _, texto in pais_ranqueados[:numero_pais]:
        for item in unique_parents.values():
            if item["text"] == texto:
                fonte = item["source"]
                pagina = item["page"]
                if pagina not in ["", None, "None"]:
                    fonte = f"{fonte} (página {pagina})"
                if fonte and fonte not in fontes:
                    fontes.append(fonte)
                break

    return contexto, fontes


# ============================================================
# CHAT
# ============================================================

def responder_pergunta(pergunta, historico):
    try:
        yield "⏳ Lendo o histórico..."

        historico_resumido = ""

        if historico:
            historico_bruto = ""
            for h in historico[-4:]:
                if isinstance(h, (list, tuple)):
                    usuario, ia = h[0], h[1]
                else:
                    usuario, ia = h.get("content", ""), ""
                historico_bruto += f"Usuário: {usuario}\nIA: {ia}\n"

            prompt_resumo = f"""
Resuma a seguinte conversa técnica em apenas um parágrafo conciso.
Mantenha apenas: entidades, intenções, regras, informações importantes.
Não responda à conversa.
CONVERSA:
{historico_bruto}
"""
            try:
                llm_resumo = Ollama(base_url=OLLAMA_URL, model=_config["summary_model"])
                historico_resumido = llm_resumo.invoke(prompt_resumo)
            except Exception:
                historico_resumido = ""

        yield "🔍 Buscando fragmentos no ChromaDB..."

        contexto, fontes = recuperar_contexto(pergunta, numero_pais=3)

        if not contexto:
            yield "Nenhum documento relevante foi encontrado."
            return

        yield f"🧠 Reranking concluído. {len(fontes)} fonte(s) relevante(s)."

        prompt_sistema = f"""
Você é um auditor especialista em leitura de editais e manuais técnicos.

Responda à pergunta do usuário baseando-se ESTRITAMENTE no contexto fornecido.

Se a resposta não estiver clara no contexto, diga explicitamente que não possui essa informação.

Não invente. Não presuma.

[MEMÓRIA DA CONVERSA]

{historico_resumido if historico_resumido else "Início da conversa."}

[CONTEXTO RECUPERADO]

{contexto}

[PERGUNTA]

{pergunta}
"""

        yield f"⚡ Enviando contexto ao modelo {_config['chat_model']}..."

        llm_principal = Ollama(base_url=OLLAMA_URL, model=_config["chat_model"])

        resposta_parcial = ""
        for chunk in llm_principal.stream(prompt_sistema):
            resposta_parcial += chunk
            yield resposta_parcial

    except Exception as e:
        traceback.print_exc()
        yield f"Erro no processamento: {str(e)}"


# ============================================================
# OLLAMA JSON
# ============================================================

def ollama_json(prompt, model, temperature=0.6, num_ctx=16384, num_predict=6000):
    url = OLLAMA_URL.rstrip("/") + "/api/generate"
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

    response = requests.post(url, json=payload, timeout=1800)
    if not response.ok:
        print(f"[Ollama Error] {response.status_code}: {response.text}")
    response.raise_for_status()

    data = response.json()
    texto = data.get("response", "")

    if not texto:
        raise RuntimeError("O Ollama retornou resposta vazia.")

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio = texto.find("{")
        fim = texto.rfind("}")
        if inicio >= 0 and fim > inicio:
            return json.loads(texto[inicio:fim + 1])
        raise RuntimeError("O modelo não retornou JSON válido.")


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


def silencio(segundos, sample_rate=24000):
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


def converter_mp3(wav_path, mp3_path):
    resultado = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", "128k", mp3_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if resultado.returncode != 0:
        raise RuntimeError("Erro no FFmpeg:\n" + resultado.stderr)
    return mp3_path


# ============================================================
# GERADOR DE DEBATE
# ============================================================

def gerar_debate(tema, duracao, estilo, voice_a, voice_b, velocidade, fazer_pegadinhas, fazer_exemplos):
    try:
        if not tema or not tema.strip():
            return None, None, "❌ Informe o tema.", ""

        yield None, None, "🔍 Consultando o ChromaDB...", ""

        duracao_int = int(duracao)
        metas_palavras = {5: 700, 10: 1400, 15: 2100, 20: 2800}
        palavras = metas_palavras.get(duracao_int, 1400)

        contexto, fontes = recuperar_contexto(tema, numero_pais=10)

        if not contexto:
            yield None, None, "❌ Nenhum documento relevante encontrado.", ""
            return

        yield (
            None, None,
            f"🧠 RAG recuperou {len(fontes)} fonte(s). Executando reranking...",
            "\n".join(f"• {x}" for x in fontes)
        )

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
            f"🧠 Gerando roteiro com {PODCAST_MODEL}...",
            "\n".join(f"• {x}" for x in fontes)
        )

        resultado = ollama_json(
            prompt=prompt, model=_config["podcast_model"], temperature=0.65,
            num_ctx=32768, num_predict=max(3000, int(palavras * 2.0))
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
            f"🎙️ Roteiro pronto: {len(dialogo_limpo)} falas.",
            "\n".join(f"• {x}" for x in fontes)
        )

        yield (
            None, None,
            f"🔊 Inicializando Kokoro em {TTS_DEVICE}...",
            "\n".join(f"• {x}" for x in fontes)
        )

        audio_array = montar_audio(
            dialogo_limpo, voice_a=voice_a, voice_b=voice_b, velocidade=float(velocidade)
        )

        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        slug      = re.sub(r"[^a-zA-Z0-9_-]+", "_", titulo)[:60]
        base      = f"{timestamp}_{slug}"
        wav_path  = os.path.join(AUDIO_PATH, f"{base}.wav")
        mp3_path  = os.path.join(AUDIO_PATH, f"{base}.mp3")
        json_path = os.path.join(AUDIO_PATH, f"{base}.json")

        sf.write(wav_path, audio_array, 24000)

        yield (
            None, None,
            "🎧 Convertendo áudio para MP3...",
            "\n".join(f"• {x}" for x in fontes)
        )

        converter_mp3(wav_path, mp3_path)

        roteiro_completo = {
            "titulo": titulo, "tema": tema, "modelo": PODCAST_MODEL,
            "voz_professor": voice_a, "voz_aluna": voice_b,
            "dialogo": dialogo_limpo, "fontes": fontes
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(roteiro_completo, f, ensure_ascii=False, indent=2)

        duracao_real = len(audio_array) / 24000
        minutos      = int(duracao_real // 60)
        segundos_val = int(duracao_real % 60)

        yield (
            mp3_path, json_path,
            f"✅ Debate concluído!\n\n🎙️ {titulo}\n⏱️ {minutos:02d}:{segundos_val:02d}\n\n"
            f"🧠 Modelo: {_config['podcast_model']}\n🔊 TTS: Kokoro\n👨 {voice_a}\n👩 {voice_b}\n",
            "\n".join(f"• {x}" for x in fontes)
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
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(AUDIO_PATH, f"teste_{voice}_{timestamp}.wav")
        sf.write(path, audio, 24000)
        return path, f"✅ Voz {voice} funcionando."
    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Erro: {str(e)}"


# ============================================================
# INTERFACE GRADIO
# ============================================================

CSS ="""
/* Altura da janela de chat */
.gradio-chatbot {
    height: 800px !important;
    min-height: 800px !important;
}

/* Tabela de documentos — altura fixa independente do conteúdo */
.tabela-documentos .table-wrap,
.tabela-documentos .scroll-hide,
.tabela-documentos > div {
    min-height: 600px !important;
    height: 600px !important;
}
"""

with gr.Blocks(title="Advanced RAG Corporativo", css=CSS) as demo:

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
Após enviar, clique em **🔄 Processar e Vetorizar** para indexar no ChromaDB.

Formatos aceitos: **PDF, TXT, DOCX**
""")

            with gr.Row():

                # ── Coluna esquerda: upload ──────────────────
                with gr.Column(scale=1):

                    gr.Markdown("### ⬆️ Enviar documentos")

                    upload_area = gr.File(
                        label="Arraste arquivos aqui ou clique em Adicionar",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".docx"],
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

                    btn_indexar = gr.Button(
                        "🔄 Processar e Vetorizar no ChromaDB",
                        variant="primary",
                        size="lg"
                    )

                    status_indexacao = gr.Textbox(
                        label="Status da vetorização",
                        lines=6,
                        interactive=False
                    )

                # ── Coluna direita: lista de documentos ──────
                with gr.Column(scale=1):

                    gr.Markdown("### 📋 Documentos na pasta")

                    tabela_docs = gr.Dataframe(
                        headers=["Arquivo", "Tamanho", "Tipo"],
                        datatype=["str", "str", "str"],
                        interactive=True,
                        label="Clique em uma linha para selecionar",
                        height=600,
                        col_count=(3, "fixed"),
                        elem_classes=["tabela-documentos"],
                    )

                    with gr.Row():

                        btn_atualizar = gr.Button(
                            "🔃 Atualizar lista",
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
                outputs=[status_indexacao]
            )

            btn_atualizar.click(
                fn=atualizar_lista_documentos,
                outputs=[tabela_docs]
            )

            btn_deletar.click(
                fn=remover_documentos,
                inputs=[tabela_docs, tabela_docs],
                outputs=[status_remocao, tabela_docs]
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
                title="Chatbot com RAG",
                chatbot=gr.Chatbot(height=600),
                stop_btn="⏹️ Interromper Geração"
            )

        # ====================================================
        # ABA DEBATE
        # ====================================================

        with gr.Tab("🎙️ Debate de Estudo"):

            gr.Markdown("""
## 🎙️ Gerar Debate de Estudo

O sistema consulta o ChromaDB, recupera os trechos mais relevantes,
utiliza o Qwen3 8B para criar o diálogo e depois gera duas vozes
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
                        choices=["pm_alex", "pm_santa"],
                        value=VOICE_A,
                        label="👨 Voz do professor"
                    )

                    voice_b = gr.Dropdown(
                        choices=["pf_dora"],
                        value=VOICE_B,
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
                choices=["pm_alex", "pm_santa", "pf_dora"],
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
                        label="📝 Modelo de Resumo do Histórico",
                        info="Modelo leve para comprimir o histórico de conversa"
                    )

                    gr.Markdown("### 🔊 Vozes Kokoro (TTS)")

                    cfg_voice_a = gr.Dropdown(
                        choices=list(VOZES_KOKORO.keys()),
                        value=_config["voice_a"],
                        label="👨 Voz do Professor",
                        info="Voz do personagem A no debate"
                    )

                    cfg_voice_b = gr.Dropdown(
                        choices=list(VOZES_KOKORO.keys()),
                        value=_config["voice_b"],
                        label="👩 Voz da Aluna",
                        info="Voz do personagem B no debate"
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

            def recarregar_e_atualizar_dropdowns():
                """
                Busca modelos atuais do Ollama e atualiza
                os três dropdowns de modelo + mantém vozes.
                """
                modelos = listar_modelos_ollama()
                chat    = _config["chat_model"]
                podcast = _config["podcast_model"]
                summary = _config["summary_model"]

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
                outputs=[status_config]
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

    print(f"Embed indexação: {EMBED_INDEXACAO_DEVICE.upper()}")
    print(f"Embed retrieval: {EMBED_RETRIEVAL_DEVICE.upper()}")
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