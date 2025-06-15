# 🤖 ZenithBot - Chatbot com Streamlit e RAG

Este é um projeto de chatbot inteligente construído com Python e Streamlit. A aplicação utiliza a técnica RAG (Retrieval-Augmented Generation) para responder a perguntas sobre um software de gestão fictício chamado "Zenith", baseando-se estritamente em um documento de texto fornecido como base de conhecimento.

O chatbot usa os modelos de Embedding e de Linguagem do Google Generative AI (Gemini) e o FAISS (Facebook AI Similarity Search) para encontrar os trechos de texto mais relevantes para a pergunta do usuário e, então, gerar uma resposta precisa.

![Imagem da interface do Chatbot](https://i.imgur.com/sBCkX9p.png)

## ✨ Funcionalidades

- **Interface de Chat Interativa:** Interface amigável e moderna construída com os novos componentes de chat do Streamlit.
- **Respostas Baseadas em Contexto (RAG):** As respostas são geradas com base em um documento específico, tornando o chatbot um especialista no assunto.
- **Busca de Similaridade:** Utiliza FAISS para uma busca vetorial de alta velocidade, encontrando os trechos de contexto mais relevantes para a pergunta do usuário.
- **Inteligência Artificial do Google:** Integrado com o `gemini-1.5-flash` para geração de texto e `text-embedding-004` para criação de embeddings.

## 🛠️ Pré-requisitos

- Python 3.8 ou superior
- Uma chave de API do [Google AI Studio](https://aistudio.google.com/app/apikey)

## 🚀 Instalação e Configuração

Siga os passos abaixo para configurar e executar o projeto em sua máquina local.

**1. Clone o repositório (ou baixe os arquivos):**
```bash
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio

2. Crie e ative um ambiente virtual:

# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

3. Instale as dependências:
Crie um arquivo chamado requirements.txt com o seguinte conteúdo:

streamlit
google-generativeai
numpy
faiss-cpu
python-dotenv

Em seguida, instale as bibliotecas:

pip install -r requirements.txt

4. Crie o arquivo de contexto:
Na raiz do projeto, crie um arquivo chamado documentos.txt e adicione o conteúdo que servirá como base de conhecimento para o chatbot.

5. Configure as variáveis de ambiente:
Crie um arquivo .env na raiz do projeto e adicione as seguintes informações:

# Substitua "SUA_API_KEY_AQUI" pela sua chave de API do Google
GENAI_API_KEY="SUA_API_KEY_AQUI"

# Caminho para o arquivo de texto com a base de conhecimento
CONTEXT_PATH="documentos.txt"

▶️ Como Executar

Com o ambiente virtual ativado e os arquivos de configuração criados, execute o seguinte comando no seu terminal:

streamlit run seu_arquivo_principal.py

Substitua seu_arquivo_principal.py pelo nome do seu script Python (ex: app.py ou usingEmbedding_py_revised.py).

A aplicação será aberta automaticamente no seu navegador padrão.
📁 Estrutura do Projeto

.
├── .venv/                  # Ambiente virtual
├── seu_arquivo_principal.py  # Código principal da aplicação Streamlit
├── documentos.txt          # Base de conhecimento do chatbot
├── .env                    # Arquivo com as variáveis de ambiente (API Key, etc.)
├── requirements.txt        # Lista de dependências Python
└── README.md               # Esta documentação

