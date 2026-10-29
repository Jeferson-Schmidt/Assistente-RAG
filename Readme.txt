Necessário para baixar os modelos diretamente para a partição do container
Volumes em ext4 no wsl2 do container para o Windows são MUITO mais rápidos do que volumes no Windows onde tem que convertes, o teste mostrou mais de 200% de performance para carregar os modelos em memória.

docker exec -it rag_ollama ollama pull qwen3:8b
docker exec -it rag_ollama ollama pull llama3.1:latest
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
|-----Docker-composse.yml
|-----App
	|---Audio
	|---Data
	|---dockerfile
	|---requirements.txt
|-----Audio
	|----Podcasts
|-----Chroma_data
|-----Data
	|---Documentos

Os arquivos que quiser utilizar como RAG, copie para a pasta data\documentos

Em Chat Rag, clique em Processar Documentos para indexar o RAG no banco vetorial ChromaDB.

Depois pode ser utilizado para perguntas no chat e para geração de audiobooks no Debate de Estudo.

Para fine-tunning dos prompts, pode ser alterado conforme quiser dentro de app.py.

Instalei o Cloudflare no meu PC.

Após a instalação executar ./cloudflared tunnel --url http://localhost:7860 e acessar via o URL que ele disponibilizar.
