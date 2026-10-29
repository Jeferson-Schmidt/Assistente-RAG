import os
import sys
import uuid
import json
import re
import traceback
import subprocess
import threading
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


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_IMPL"] = (
    "chromadb.telemetry.dummy.DummyTelemetry"
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

CHROMA_HOST = os.getenv(
    "CHROMA_HOST",
    "localhost"
)

CHROMA_PORT = int(
    os.getenv("CHROMA_PORT", "8000")
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "llama3.1:latest"
)

SUMMARY_MODEL = os.getenv(
    "SUMMARY_MODEL",
    "llama3.2:1b"
)

PODCAST_MODEL = os.getenv(
    "PODCAST_MODEL",
    "qwen3:8b"
)

KOKORO_DEVICE = os.getenv(
    "KOKORO_DEVICE",
    "cuda"
)

VOICE_A = os.getenv(
    "KOKORO_VOICE_A",
    "pm_alex"
)

VOICE_B = os.getenv(
    "KOKORO_VOICE_B",
    "pf_dora"
)

DOCUMENTS_PATH = "/app/data/documentos"

AUDIO_PATH = "/app/audio/podcasts"

COLLECTION_NAME = "edital_avancado"


# ============================================================
# CRIA DIRETÓRIOS
# ============================================================

os.makedirs(DOCUMENTS_PATH, exist_ok=True)
os.makedirs(AUDIO_PATH, exist_ok=True)


# ============================================================
# CONTROLE DE GPU / KOKORO
# ============================================================

if KOKORO_DEVICE == "cuda" and torch.cuda.is_available():
    TTS_DEVICE = "cuda"
else:
    TTS_DEVICE = "cpu"

print("=" * 70)
print("SISTEMA RAG + PODCAST")
print("=" * 70)
print(f"Ollama:       {OLLAMA_URL}")
print(f"Chat model:   {CHAT_MODEL}")
print(f"Podcast LLM:  {PODCAST_MODEL}")
print(f"TTS device:   {TTS_DEVICE}")
print(f"Voice A:      {VOICE_A}")
print(f"Voice B:      {VOICE_B}")
print("=" * 70)


# ============================================================
# MODELOS DE EMBEDDING / RERANKING
# ============================================================

print("Carregando embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu"
    }
)

print("Carregando CrossEncoder...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu"
)


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

        print(
            f"Inicializando Kokoro em {TTS_DEVICE}..."
        )

        try:

            kokoro_pipeline = KPipeline(
                lang_code="p",
                device=TTS_DEVICE
            )

            print("Kokoro inicializado com sucesso.")

            return kokoro_pipeline

        except Exception as e:

            print(
                "ERRO AO INICIALIZAR KOKORO:"
            )

            traceback.print_exc()

            raise RuntimeError(
                f"Não foi possível inicializar o Kokoro: {e}"
            )


# ============================================================
# CHROMA CLIENT
# ============================================================

def criar_cliente_chroma():

    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(
            anonymized_telemetry=False
        )
    )


# ============================================================
# LISTA DOCUMENTOS
# ============================================================

def listar_documentos():

    if not os.path.exists(DOCUMENTS_PATH):
        return []

    arquivos = []

    for arquivo in Path(DOCUMENTS_PATH).iterdir():

        if arquivo.is_file():

            extensao = arquivo.suffix.lower()

            if extensao in [
                ".pdf",
                ".txt",
                ".docx"
            ]:
                arquivos.append(arquivo.name)

    return sorted(arquivos)


# ============================================================
# INDEXAÇÃO
# ============================================================

def indexar_documentos():

    try:

        if (
            not os.path.exists(DOCUMENTS_PATH)
            or not os.listdir(DOCUMENTS_PATH)
        ):

            return (
                "Nenhum arquivo encontrado em "
                "data/documentos."
            )

        pdf_loader = PyPDFDirectoryLoader(
            DOCUMENTS_PATH
        )

        txt_loader = DirectoryLoader(
            DOCUMENTS_PATH,
            glob="*.txt",
            loader_cls=TextLoader
        )

        docx_loader = DirectoryLoader(
            DOCUMENTS_PATH,
            glob="*.docx",
            loader_cls=Docx2txtLoader
        )

        docs = []

        docs.extend(
            pdf_loader.load()
        )

        docs.extend(
            txt_loader.load()
        )

        docs.extend(
            docx_loader.load()
        )

        if not docs:

            return (
                "Nenhum documento válido encontrado."
            )

        # ----------------------------------------------------
        # PARENT SPLITTER
        # ----------------------------------------------------

        parent_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=2500,
                chunk_overlap=200
            )
        )

        # ----------------------------------------------------
        # CHILD SPLITTERS
        # ----------------------------------------------------

        child_splitters = [

            RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"],
                chunk_size=300,
                chunk_overlap=50
            ),

            RecursiveCharacterTextSplitter(
                separators=[
                    ". ",
                    "? ",
                    "! ",
                    "; "
                ],
                chunk_size=300,
                chunk_overlap=50
            ),

            RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=40
            ),

            RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=80
            ),

            RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=160
            )
        ]

        all_child_docs = []

        parents = parent_splitter.split_documents(
            docs
        )

        # ----------------------------------------------------
        # GERA FILHOS
        # ----------------------------------------------------

        for parent in parents:

            parent_id = str(
                uuid.uuid4()
            )

            parent_text = (
                parent.page_content
            )

            original_source = (
                parent.metadata.get(
                    "source",
                    ""
                )
            )

            source_name = os.path.basename(
                original_source
            )

            page_number = (
                parent.metadata.get(
                    "page",
                    ""
                )
            )

            for splitter in child_splitters:

                children_texts = (
                    splitter.split_text(
                        parent_text
                    )
                )

                for child_text in children_texts:

                    if (
                        len(
                            child_text.strip()
                        ) <= 10
                    ):
                        continue

                    metadata = {

                        "parent_id":
                            parent_id,

                        "parent_text":
                            parent_text,

                        "source":
                            source_name,

                        "page":
                            str(page_number)
                    }

                    all_child_docs.append(
                        Document(
                            page_content=child_text,
                            metadata=metadata
                        )
                    )

        # ----------------------------------------------------
        # CHROMA
        # ----------------------------------------------------

        client = criar_cliente_chroma()

        try:

            client.delete_collection(
                name=COLLECTION_NAME
            )

        except Exception:

            pass

        Chroma.from_documents(

            documents=all_child_docs,

            embedding=embeddings,

            client=client,

            collection_name=COLLECTION_NAME
        )

        return (
            f"Sucesso!\n\n"
            f"Documentos carregados: "
            f"{len(docs)}\n\n"
            f"Blocos pais: "
            f"{len(parents)}\n\n"
            f"Fragmentos filhos: "
            f"{len(all_child_docs)}"
        )

    except Exception as e:

        traceback.print_exc()

        return (
            f"Erro na indexação:\n{str(e)}"
        )


# ============================================================
# RAG - RECUPERA CONTEXTO
# ============================================================

def recuperar_contexto(
    pergunta,
    numero_pais=3,
    filtro_documento=None
):

    client = criar_cliente_chroma()

    vectordb = Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # --------------------------------------------------------
    # BUSCA
    # --------------------------------------------------------

    if filtro_documento:

        try:

            docs_filhos = (
                vectordb.similarity_search(
                    pergunta,
                    k=20,
                    filter={
                        "source":
                            filtro_documento
                    }
                )
            )

        except Exception:

            docs_filhos = (
                vectordb.similarity_search(
                    pergunta,
                    k=20
                )
            )

    else:

        docs_filhos = (
            vectordb.similarity_search(
                pergunta,
                k=20
            )
        )

    # --------------------------------------------------------
    # RECUPERA PAIS ÚNICOS
    # --------------------------------------------------------

    unique_parents = {}

    for doc in docs_filhos:

        pid = doc.metadata.get(
            "parent_id"
        )

        if pid and pid not in unique_parents:

            unique_parents[pid] = {

                "text":
                    doc.metadata.get(
                        "parent_text",
                        doc.page_content
                    ),

                "source":
                    doc.metadata.get(
                        "source",
                        ""
                    ),

                "page":
                    doc.metadata.get(
                        "page",
                        ""
                    )
            }

    if not unique_parents:

        return None, []

    # --------------------------------------------------------
    # RERANKING
    # --------------------------------------------------------

    candidatos = [
        item["text"]
        for item in unique_parents.values()
    ]

    pares_avaliacao = [
        [pergunta, texto]
        for texto in candidatos
    ]

    notas = reranker.predict(
        pares_avaliacao
    )

    pais_ranqueados = sorted(
        zip(
            notas,
            candidatos
        ),
        key=lambda x: x[0],
        reverse=True
    )

    top_textos = [
        texto
        for nota, texto
        in pais_ranqueados[
            :numero_pais
        ]
    ]

    contexto = (
        "\n\n"
        "--- FIM DO TRECHO / NOVO TRECHO ---"
        "\n\n"
    ).join(top_textos)

    # --------------------------------------------------------
    # FONTES
    # --------------------------------------------------------

    fontes = []

    for nota, texto in pais_ranqueados[
        :numero_pais
    ]:

        for item in unique_parents.values():

            if item["text"] == texto:

                fonte = item["source"]

                pagina = item["page"]

                if pagina not in [
                    "",
                    None,
                    "None"
                ]:

                    fonte = (
                        f"{fonte} "
                        f"(página {pagina})"
                    )

                if fonte and fonte not in fontes:

                    fontes.append(fonte)

                break

    return contexto, fontes


# ============================================================
# CHAT NORMAL
# ============================================================

def responder_pergunta(
    pergunta,
    historico
):

    try:

        yield (
            "⏳ Lendo o histórico..."
        )

        historico_resumido = ""

        if historico:

            historico_bruto = ""

            for h in historico[-4:]:

                if isinstance(h, (list, tuple)):

                    usuario = h[0]
                    ia = h[1]

                else:

                    usuario = h.get(
                        "content",
                        ""
                    )

                    ia = ""

                historico_bruto += (
                    f"Usuário: {usuario}\n"
                    f"IA: {ia}\n"
                )

            prompt_resumo = f"""

Resuma a seguinte conversa técnica
em apenas um parágrafo conciso.

Mantenha apenas:

- entidades
- intenções
- regras
- informações importantes

Não responda à conversa.

CONVERSA:

{historico_bruto}
"""

            try:

                llm_resumo = Ollama(
                    base_url=OLLAMA_URL,
                    model=SUMMARY_MODEL
                )

                historico_resumido = (
                    llm_resumo.invoke(
                        prompt_resumo
                    )
                )

            except Exception:

                historico_resumido = ""

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        yield (
            "🔍 Buscando fragmentos no ChromaDB..."
        )

        contexto, fontes = (
            recuperar_contexto(
                pergunta,
                numero_pais=3
            )
        )

        if not contexto:

            yield (
                "Nenhum documento relevante "
                "foi encontrado."
            )

            return

        yield (
            f"🧠 Reranking concluído. "
            f"{len(fontes)} fonte(s) relevante(s)."
        )

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt_sistema = f"""

Você é um auditor especialista
em leitura de editais e manuais técnicos.

Responda à pergunta do usuário
baseando-se ESTRITAMENTE no contexto
fornecido.

Se a resposta não estiver clara no
contexto, diga explicitamente que
não possui essa informação.

Não invente.

Não presuma.

[MEMÓRIA DA CONVERSA]

{
    historico_resumido
    if historico_resumido
    else "Início da conversa."
}

[CONTEXTO RECUPERADO]

{contexto}

[PERGUNTA]

{pergunta}
"""

        yield (
            "⚡ Enviando contexto para a RTX 5070..."
        )

        llm_principal = Ollama(
            base_url=OLLAMA_URL,
            model=CHAT_MODEL
        )

        resposta_parcial = ""

        for chunk in llm_principal.stream(
            prompt_sistema
        ):

            resposta_parcial += chunk

            yield resposta_parcial

    except Exception as e:

        traceback.print_exc()

        yield (
            f"Erro no processamento: {str(e)}"
        )


# ============================================================
# OLLAMA - GERA JSON
# ============================================================

def ollama_json(
    prompt,
    model,
    temperature=0.6,
    num_ctx=16384,
    num_predict=6000
):

    url = (
        OLLAMA_URL.rstrip("/")
        + "/api/generate"
    )

    payload = {

        "model": model,

        "prompt": prompt,

        "stream": False,

        "format": "json",

        "think": False,

        "options": {

            "temperature":
                temperature,

            "top_p":
                0.9,

            "num_ctx":
                num_ctx,

            "num_predict":
                num_predict,

            "repeat_penalty":
                1.05
        }
    }

    response = requests.post(
        url,
        json=payload,
        timeout=1800
    )

    response.raise_for_status()

    data = response.json()

    texto = data.get(
        "response",
        ""
    )

    if not texto:

        raise RuntimeError(
            "O Ollama retornou resposta vazia."
        )

    try:

        return json.loads(texto)

    except json.JSONDecodeError:

        # Tentativa de encontrar JSON
        # dentro da resposta.

        inicio = texto.find("{")
        fim = texto.rfind("}")

        if inicio >= 0 and fim > inicio:

            trecho = texto[
                inicio:fim + 1
            ]

            return json.loads(trecho)

        raise RuntimeError(
            "O modelo não retornou JSON válido."
        )


# ============================================================
# LIMPEZA DO TEXTO PARA TTS
# ============================================================

def limpar_texto_tts(texto):

    if not texto:
        return ""

    texto = str(texto)

    # Remove markdown
    texto = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        texto
    )

    texto = re.sub(
        r"\*(.*?)\*",
        r"\1",
        texto
    )

    texto = re.sub(
        r"`(.*?)`",
        r"\1",
        texto
    )

    texto = texto.replace(
        "\n",
        " "
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    )

    return texto.strip()


# ============================================================
# TTS
# ============================================================

def gerar_fala(
    texto,
    voice,
    velocidade=1.0
):

    pipeline = (
        inicializar_kokoro()
    )

    texto = limpar_texto_tts(
        texto
    )

    if not texto:

        return np.zeros(
            2400,
            dtype=np.float32
        )

    partes = []

    with kokoro_lock:

        generator = pipeline(
            texto,
            voice=voice,
            speed=velocidade,
            split_pattern=r"\n+"
        )

        for _, _, audio in generator:

            if audio is not None:

                partes.append(
                    np.asarray(
                        audio,
                        dtype=np.float32
                    )
                )

    if not partes:

        raise RuntimeError(
            f"Kokoro não gerou áudio "
            f"para a voz {voice}."
        )

    return np.concatenate(
        partes
    )


# ============================================================
# SILÊNCIO
# ============================================================

def silencio(
    segundos,
    sample_rate=24000
):

    quantidade = int(
        segundos * sample_rate
    )

    return np.zeros(
        quantidade,
        dtype=np.float32
    )


# ============================================================
# MONTA ÁUDIO COMPLETO
# ============================================================

def montar_audio(
    dialogo,
    voice_a,
    voice_b,
    velocidade=1.0
):

    audio_partes = []

    for item in dialogo:

        speaker = str(
            item.get(
                "speaker",
                ""
            )
        ).lower()

        texto = item.get(
            "text",
            ""
        )

        if not texto:
            continue

        if speaker in [
            "professor",
            "a",
            "speaker_a"
        ]:

            voice = voice_a

        else:

            voice = voice_b

        audio = gerar_fala(
            texto,
            voice,
            velocidade
        )

        audio_partes.append(
            audio
        )

        # Pequena pausa entre falas
        audio_partes.append(
            silencio(0.20)
        )

    if not audio_partes:

        raise RuntimeError(
            "Nenhuma fala foi gerada."
        )

    return np.concatenate(
        audio_partes
    )


# ============================================================
# CONVERSÃO WAV -> MP3
# ============================================================

def converter_mp3(
    wav_path,
    mp3_path
):

    comando = [

        "ffmpeg",

        "-y",

        "-i",
        wav_path,

        "-codec:a",
        "libmp3lame",

        "-b:a",
        "128k",

        mp3_path
    ]

    resultado = subprocess.run(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if resultado.returncode != 0:

        raise RuntimeError(
            "Erro no FFmpeg:\n"
            + resultado.stderr
        )

    return mp3_path


# ============================================================
# GERADOR DE DEBATE
# ============================================================

def gerar_debate(
    tema,
    duracao,
    estilo,
    voice_a,
    voice_b,
    velocidade,
    fazer_pegadinhas,
    fazer_exemplos
):

    try:

        if not tema or not tema.strip():

            return (
                None,
                None,
                "❌ Informe o tema.",
                ""
            )

        yield (
            None,
            None,
            "🔍 Consultando o ChromaDB...",
            ""
        )

        # ----------------------------------------------------
        # DURAÇÃO
        # ----------------------------------------------------

        duracao_int = int(
            duracao
        )

        # Aproximadamente 135-150 palavras/minuto
        metas_palavras = {

            5: 700,

            10: 1400,

            15: 2100,

            20: 2800
        }

        palavras = metas_palavras.get(
            duracao_int,
            1400
        )

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        contexto, fontes = (
            recuperar_contexto(
                tema,
                numero_pais=6
            )
        )

        if not contexto:

            yield (
                None,
                None,
                "❌ Nenhum documento relevante encontrado.",
                ""
            )

            return

        yield (
            None,
            None,
            (
                f"🧠 RAG recuperou "
                f"{len(fontes)} fonte(s). "
                f"Executando reranking..."
            ),
            "\n".join(
                f"• {x}"
                for x in fontes
            )
        )

        # ----------------------------------------------------
        # REGRAS OPCIONAIS
        # ----------------------------------------------------

        pegadinhas = (
            "Inclua possíveis pegadinhas "
            "de concurso quando existirem "
            "no contexto."
            if fazer_pegadinhas
            else
            "Não force pegadinhas artificiais."
        )

        exemplos = (
            "Inclua exemplos práticos "
            "para facilitar a compreensão."
            if fazer_exemplos
            else
            "Use exemplos apenas quando "
            "forem realmente úteis."
        )

        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = f"""

Você é um roteirista especializado
em preparação para concursos públicos
brasileiros.

Você está criando um episódio de estudo
baseado exclusivamente nos documentos
recuperados pelo sistema RAG.

TEMA:

{tema}

OBJETIVO:

Criar um debate didático entre duas pessoas.

PERSONAGEM A:

Professor especialista.

Características:

- domínio técnico;
- didático;
- paciente;
- objetivo;
- corrige erros;
- explica conceitos;
- relaciona conceitos quando isso
  estiver presente no contexto.

PERSONAGEM B:

Aluna inteligente e questionadora.

Características:

- faz perguntas;
- demonstra dúvidas plausíveis;
- apresenta interpretações alternativas;
- tenta encontrar exceções;
- questiona pontos difíceis;
- não deve parecer artificial.

ESTILO:

{estilo}

REGRAS ABSOLUTAS:

1. Use somente informações presentes
   no CONTEXTO.

2. Não invente leis, artigos, números,
   conceitos ou exemplos que não estejam
   sustentados pelo contexto.

3. Se alguma informação importante
   não estiver no contexto, não invente.

4. O diálogo deve ensinar.

5. Evite transformar o episódio
   em uma palestra.

6. A conversa deve parecer natural.

7. Alterne entre professor e aluna.

8. Não faça respostas de apenas uma frase
   o tempo todo.

9. Evite repetição.

10. Priorize conteúdo relevante para
    concursos públicos.

11. {pegadinhas}

12. {exemplos}

TAMANHO:

Produza aproximadamente
{palavras} palavras.

FORMATO:

Retorne EXCLUSIVAMENTE um JSON válido.

Estrutura obrigatória:

{{
  "titulo": "Título do episódio",
  "dialogo": [
    {{
      "speaker": "professor",
      "text": "Fala do professor"
    }},
    {{
      "speaker": "aluna",
      "text": "Fala da aluna"
    }}
  ]
}}

Não coloque markdown.

Não coloque ```json.

Não coloque comentários.

CONTEXTO DOS DOCUMENTOS:

{contexto}
"""

        yield (
            None,
            None,
            (
                f"🧠 Gerando roteiro com "
                f"{PODCAST_MODEL}..."
            ),
            "\n".join(
                f"• {x}"
                for x in fontes
            )
        )

        # ----------------------------------------------------
        # GERA JSON
        # ----------------------------------------------------

        resultado = ollama_json(
            prompt=prompt,
            model=PODCAST_MODEL,
            temperature=0.65,
            num_ctx=32768,
            num_predict=max(
                3000,
                int(
                    palavras * 2.0
                )
            )
        )

        titulo = resultado.get(
            "titulo",
            "Debate de Estudo"
        )

        dialogo = resultado.get(
            "dialogo",
            []
        )

        if not dialogo:

            raise RuntimeError(
                "O modelo não retornou diálogo."
            )

        # ----------------------------------------------------
        # LIMPA DIÁLOGO
        # ----------------------------------------------------

        dialogo_limpo = []

        for item in dialogo:

            if not isinstance(
                item,
                dict
            ):
                continue

            speaker = item.get(
                "speaker",
                ""
            )

            text = item.get(
                "text",
                ""
            )

            if not text:
                continue

            if speaker not in [
                "professor",
                "aluna",
                "a",
                "b",
                "speaker_a",
                "speaker_b"
            ]:

                continue

            dialogo_limpo.append({

                "speaker":
                    speaker,

                "text":
                    limpar_texto_tts(
                        text
                    )
            })

        if not dialogo_limpo:

            raise RuntimeError(
                "O diálogo retornado "
                "não possui falas válidas."
            )

        yield (
            None,
            None,
            (
                f"🎙️ Roteiro pronto: "
                f"{len(dialogo_limpo)} falas."
            ),
            "\n".join(
                f"• {x}"
                for x in fontes
            )
        )

        # ----------------------------------------------------
        # TTS
        # ----------------------------------------------------

        yield (
            None,
            None,
            (
                "🔊 Inicializando "
                f"Kokoro em {TTS_DEVICE}..."
            ),
            "\n".join(
                f"• {x}"
                for x in fontes
            )
        )

        audio_array = montar_audio(
            dialogo_limpo,
            voice_a=voice_a,
            voice_b=voice_b,
            velocidade=float(
                velocidade
            )
        )

        # ----------------------------------------------------
        # ARQUIVOS
        # ----------------------------------------------------

        timestamp = (
            __import__("datetime")
            .datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        slug = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            titulo
        )[:60]

        base = (
            f"{timestamp}_{slug}"
        )

        wav_path = os.path.join(
            AUDIO_PATH,
            f"{base}.wav"
        )

        mp3_path = os.path.join(
            AUDIO_PATH,
            f"{base}.mp3"
        )

        json_path = os.path.join(
            AUDIO_PATH,
            f"{base}.json"
        )

        # ----------------------------------------------------
        # WAV
        # ----------------------------------------------------

        sf.write(
            wav_path,
            audio_array,
            24000
        )

        yield (
            None,
            None,
            "🎧 Convertendo áudio para MP3...",
            "\n".join(
                f"• {x}"
                for x in fontes
            )
        )

        # ----------------------------------------------------
        # MP3
        # ----------------------------------------------------

        converter_mp3(
            wav_path,
            mp3_path
        )

        # ----------------------------------------------------
        # SALVA ROTEIRO
        # ----------------------------------------------------

        roteiro_completo = {

            "titulo":
                titulo,

            "tema":
                tema,

            "modelo":
                PODCAST_MODEL,

            "voz_professor":
                voice_a,

            "voz_aluna":
                voice_b,

            "dialogo":
                dialogo_limpo,

            "fontes":
                fontes
        }

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as arquivo:

            json.dump(
                roteiro_completo,
                arquivo,
                ensure_ascii=False,
                indent=2
            )

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        duracao_real = (
            len(audio_array)
            / 24000
        )

        minutos = int(
            duracao_real // 60
        )

        segundos = int(
            duracao_real % 60
        )

        status_final = (

            f"✅ Debate concluído!\n\n"

            f"🎙️ {titulo}\n"

            f"⏱️ {minutos:02d}:"
            f"{segundos:02d}\n\n"

            f"🧠 Modelo: "
            f"{PODCAST_MODEL}\n"

            f"🔊 TTS: Kokoro\n"

            f"👨 {voice_a}\n"

            f"👩 {voice_b}\n"
        )

        yield (
            mp3_path,
            json_path,
            status_final,
            "\n".join(
                f"• {x}"
                for x in fontes
            )
        )

    except Exception as e:

        traceback.print_exc()

        yield (
            None,
            None,
            (
                "❌ ERRO AO GERAR DEBATE\n\n"
                + str(e)
            ),
            ""
        )


# ============================================================
# TESTE DE VOZ
# ============================================================

def testar_voz(voice):

    try:

        texto = (
            "Olá. Este é um teste de voz "
            "do seu sistema local de estudos. "
            "Estamos utilizando o Kokoro para "
            "gerar áudio em português brasileiro."
        )

        audio = gerar_fala(
            texto,
            voice
        )

        timestamp = (
            __import__("datetime")
            .datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        path = os.path.join(
            AUDIO_PATH,
            f"teste_{voice}_{timestamp}.wav"
        )

        sf.write(
            path,
            audio,
            24000
        )

        return (
            path,
            f"✅ Voz {voice} funcionando."
        )

    except Exception as e:

        traceback.print_exc()

        return (
            None,
            f"❌ Erro: {str(e)}"
        )


# ============================================================
# INTERFACE GRADIO
# ============================================================

with gr.Blocks(
    title="Advanced RAG Corporativo"
) as demo:

    gr.Markdown(
        """
# 🚀 Advanced RAG Corporativo

### RAG local + Ollama + ChromaDB + Podcast de Estudo

Sistema totalmente local para consulta de documentos
e geração de debates em áudio.
"""
    )

    with gr.Tabs():

        # ====================================================
        # ABA CHAT
        # ====================================================

        with gr.Tab(
            "💬 Chat RAG"
        ):

            gr.Markdown(
                """
Faça perguntas sobre os documentos
indexados no ChromaDB.
"""
            )

            with gr.Row():

                with gr.Column(
                    scale=1
                ):

                    btn_indexar = gr.Button(
                        "🔄 Processar Documentos",
                        variant="primary"
                    )

                    status_output = gr.Textbox(
                        label="Status da Indexação",
                        placeholder=(
                            "Aguardando ação..."
                        ),
                        lines=8
                    )

                with gr.Column(
                    scale=3
                ):

                    chatbot = gr.ChatInterface(
                        fn=responder_pergunta,
                        title="Auditor de Editais",
                        stop_btn="⏹️ Interromper Geração"
                    )

            btn_indexar.click(
                fn=indexar_documentos,
                outputs=status_output
            )


        # ====================================================
        # ABA DEBATE
        # ====================================================

        with gr.Tab(
            "🎙️ Debate de Estudo"
        ):

            gr.Markdown(
                """
## 🎙️ Gerar Debate de Estudo

O sistema consulta o ChromaDB, recupera os
trechos mais relevantes, utiliza o Qwen3 8B
para criar o diálogo e depois gera duas vozes
em português brasileiro com o Kokoro.
"""
            )

            with gr.Row():

                with gr.Column(
                    scale=1
                ):

                    tema = gr.Textbox(
                        label="📚 Tema",
                        placeholder=(
                            "Ex.: "
                            "Não cumulatividade do ICMS"
                        ),
                        lines=3
                    )

                    duracao = gr.Dropdown(
                        choices=[
                            5,
                            10,
                            15,
                            20
                        ],
                        value=10,
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

                        choices=[

                            "pm_alex",
                            "pm_santa"
                        ],

                        value=VOICE_A,

                        label=(
                            "👨 Voz do professor"
                        )
                    )

                    voice_b = gr.Dropdown(

                        choices=[
                            "pf_dora"
                        ],

                        value=VOICE_B,

                        label=(
                            "👩 Voz da aluna"
                        )
                    )

                    velocidade = gr.Slider(

                        minimum=0.85,

                        maximum=1.15,

                        value=1.0,

                        step=0.05,

                        label=(
                            "🎚️ Velocidade da fala"
                        )
                    )

                    fazer_pegadinhas = gr.Checkbox(

                        value=True,

                        label=(
                            "🪤 Explorar pegadinhas "
                            "de concurso"
                        )
                    )

                    fazer_exemplos = gr.Checkbox(

                        value=True,

                        label=(
                            "💡 Utilizar exemplos"
                        )
                    )

                    gerar_btn = gr.Button(

                        "🎙️ GERAR DEBATE",

                        variant="primary",

                        size="lg"
                    )

                with gr.Column(
                    scale=2
                ):

                    audio_output = gr.Audio(

                        label="🎧 Podcast",

                        format="mp3",

                        autoplay=False,

                        show_download_button=True
                    )

                    roteiro_download = gr.File(

                        label=(
                            "📄 Baixar roteiro JSON"
                        )
                    )

                    debate_status = gr.Textbox(

                        label=(
                            "⚙️ Status do processamento"
                        ),

                        lines=8
                    )

                    fontes_output = gr.Textbox(

                        label=(
                            "📚 Fontes utilizadas"
                        ),

                        lines=8
                    )

            gerar_btn.click(

                fn=gerar_debate,

                inputs=[

                    tema,

                    duracao,

                    estilo,

                    voice_a,

                    voice_b,

                    velocidade,

                    fazer_pegadinhas,

                    fazer_exemplos
                ],

                outputs=[

                    audio_output,

                    roteiro_download,

                    debate_status,

                    fontes_output
                ]
            )


        # ====================================================
        # ABA TESTE DE VOZ
        # ====================================================

        with gr.Tab(
            "🔊 Testar Vozes"
        ):

            gr.Markdown(
                """
Use esta aba para testar o Kokoro antes de
gerar um episódio completo.
"""
            )

            voice_test = gr.Dropdown(

                choices=[

                    "pm_alex",
                    "pm_santa",
                    "pf_dora"
                ],

                value="pm_alex",

                label="Voz"
            )

            test_btn = gr.Button(
                "🔊 Gerar Teste"
            )

            test_audio = gr.Audio(
                label="Resultado",
                format="wav"
            )

            test_status = gr.Textbox(
                label="Status"
            )

            test_btn.click(

                fn=testar_voz,

                inputs=voice_test,

                outputs=[
                    test_audio,
                    test_status
                ]
            )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("INICIANDO ADVANCED RAG")
    print("=" * 70)

    print(
        f"GPU disponível: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"VRAM total: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(
        f"TTS device: {TTS_DEVICE}"
    )

    print("=" * 70)

    demo.queue(
        max_size=2
    )

    demo.launch(

        server_name="0.0.0.0",

        server_port=7860,

        share=False,

        prevent_thread_lock=False
    )