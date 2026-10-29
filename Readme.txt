Necessário para baixar os modelos diretamente para a partição do container
Volumes em ext4 no wsl2 do container para o Windows são MUITO mais rápidos do que volumes no Windows onde tem que convertes, o teste mostrou mais de 200% de performance para carregar os modelos em memória.

docker exec -it rag_ollama ollama pull qwen3:8b
docker exec -it rag_ollama ollama pull llama3.2:1b

Se quiser testar outros modelos maiores, de acordo com o seu hardware, apenas baixe eles com os comandos acima, altere no docker-compose e no app.py.

Depois de os modelos serem baixados, execute:

docker compose up -d --build

Se precisar alterar algo no dockerfile, não precisa dar build em tudo de novo, execute apenas

docker compose build --no-cache rag_gradio_app

se alterar algo no app.py ou no docker-compose, apenas execute

Docker compose down
Docker compose up -d

E para subir o programa quando quiser, execute:

Docker compose up -d

Para acessar o sistema, abra o browser(Chrome, Firefox, etc...) e digite:

http://localhost:7860/

Verificar logs:

docker logs -f rag_gradio_app --tail 50
docker logs -f rag_ollama --tail 50

Estrutura de diretório:

/MAIN
|
|-----docker-compose.yml
|-----app
	|---app.py
	|---dockerfile
	|---requirements.txt
|-----audio
	|----podcasts        (MP3 + roteiro JSON de cada debate)
|-----data
	|---documentos       (arquivos usados no RAG)

Volumes Docker (ext4 no WSL2, mais rápidos):
- ollama_storage  -> modelos do Ollama
- hf_cache        -> modelos de embedding, reranker e Kokoro
- chroma_storage  -> banco vetorial ChromaDB
- rag_index       -> SQLite com os blocos pais e o controle de arquivos indexados

A pasta antiga chroma_data/ não é mais usada e pode ser apagada.

Os arquivos que quiser utilizar como RAG, envie pela aba Documentos (ou copie para data\documentos).

Na aba Documentos, clique em Processar e Vetorizar. A indexação é incremental:
só arquivos novos ou alterados são processados, e arquivos removidos saem do índice.
Marque "Reindexar tudo do zero" se trocar o modelo de embedding ou a estratégia de chunking.

Para remover documentos, clique nas linhas da tabela para selecioná-los e depois em Remover selecionados
(remove o arquivo e os fragmentos dele no índice).

Depois pode ser utilizado para perguntas no chat e para geração de audiobooks no Debate de Estudo.

Para fine-tunning dos prompts, pode ser alterado conforme quiser dentro de app.py.

VRAM: embedding e reranker rodam em fp16 na GPU (~2,4 GB). Se usar um modelo maior no Ollama
e faltar VRAM, mude EMBED_DEVICE e/ou RERANK_DEVICE para cpu no docker-compose.yml.
Confira com: docker exec rag_ollama ollama ps  (o ideal é "100% GPU").

Instalei o Cloudflare no meu PC.

Após a instalação executar ./cloudflared tunnel --url http://localhost:7860 e acessar via o URL que ele disponibilizar.
