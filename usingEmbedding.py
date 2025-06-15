import os
import time
import streamlit as st
import google.generativeai as genai
import numpy as np
import faiss
from dotenv import load_dotenv

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(
    page_title="ZenithBot - O seu assistente",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()

# --- CONSTANTES E MODELOS ---

# Prompt do sistema para guiar o modelo de linguagem
PROMPT_TEMPLATE = """
Você é o ZenithBot, um assistente especializado no software de gestão Zenith.
Sua função é responder perguntas baseando-se estritamente no contexto fornecido.
Seja claro, objetivo e use apenas as informações disponíveis.

Se a resposta não estiver no contexto, responda:
"Desculpe, não encontrei essa informação nos meus documentos."

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}
"""

EMBEDDING_MODEL = 'text-embedding-004'

# --- FUNÇÕES PRINCIPAIS ---

def configure_genai():
    """Configura a API do Google Generative AI com a chave fornecida."""
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        st.error("A variável de ambiente GENAI_API_KEY não foi definida. Por favor, configure-a no seu arquivo .env.")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def split_text_into_chunks(text: str) -> list[str]:
    """Divide um texto longo em pedaços menores (chunks) usando a quebra de linha."""
    parts = text.split("\n")
    # Remove espaços em branco e garante que chunks vazios não sejam incluídos
    chunks = [part.strip() for part in parts if part.strip()]
    return chunks

@st.cache_resource(show_spinner="Analisando o documento e preparando a base de conhecimento...")
def create_faiss_index():
    """
    Lê o documento de contexto, gera embeddings para os chunks de texto
    e cria um índice FAISS para busca rápida de similaridade.
    """
    context_path = os.getenv('CONTEXT_PATH')

    if not context_path or not os.path.exists(context_path):
        st.error(
            f"Arquivo de contexto não encontrado. "
            f"Verifique se a variável 'CONTEXT_PATH' no seu arquivo .env aponta para um arquivo válido."
        )
        st.stop()

    with open(context_path, 'r', encoding='utf-8') as f:
        document_text = f.read()

    text_chunks = split_text_into_chunks(document_text)

    # Gera embeddings para os chunks de texto
    response = genai.embed_content(model=EMBEDDING_MODEL, content=text_chunks)
    embeddings = np.array(response['embedding'], dtype='float32')

    # Normaliza os vetores para usar similaridade de cosseno (com produto interno)
    faiss.normalize_L2(embeddings)

    # Cria o índice FAISS
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    return index, text_chunks

def retrieve_relevant_contexts(query: str, index: faiss.Index, text_chunks: list, top_k: int = 3) -> list[str]:
    """Busca os chunks de texto mais relevantes para a pergunta do usuário no índice FAISS."""
    query_embedding_response = genai.embed_content(model=EMBEDDING_MODEL, content=query)
    query_vector = np.array(query_embedding_response['embedding'], dtype='float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)

    _, indices = index.search(query_vector, top_k)
    
    # Retorna os textos correspondentes, ignorando resultados inválidos (-1)
    return [text_chunks[i] for i in indices[0] if i != -1]

def generate_response_with_llm(prompt: str, llm_model) -> str:
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Ocorreu um erro ao gerar a resposta: {e}")
        return "Desculpe, tive um problema ao processar sua solicitação."

def simulate_typing_effect(text: str):
    """Exibe o texto com um efeito de digitação e um cursor."""
    message_placeholder = st.empty()
    full_response = ""
    for chunk in text.split():
        full_response += chunk + " "
        time.sleep(0.05)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response.strip())


# --- INTERFACE DO STREAMLIT ---

st.title("🤖 ZenithBot - Assistente Virtual")
st.caption("Faça uma pergunta sobre o software de gestão Zenith e eu responderei com base no meu conhecimento.")

# Configura o modelo e cria o índice (executado apenas uma vez)
llm_model = configure_genai()
faiss_index, base_texts = create_faiss_index()

# Inicializa o histórico de chat no estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Como posso ajudar você a entender o Zenith hoje?"}]

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a entrada do usuário
if user_input := st.chat_input("Qual é a sua dúvida?"):
    # Adiciona e exibe a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Processa a resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # 1. Recupera contextos relevantes
            relevant_contexts = retrieve_relevant_contexts(user_input, faiss_index, base_texts)
            
            # 2. Constrói o prompt para o LLM
            if relevant_contexts:
                context_str = "\n\n---\n\n".join(relevant_contexts)
                final_prompt = PROMPT_TEMPLATE.format(contexto=context_str, pergunta=user_input)
            else:
                final_prompt = user_input

            # 3. Gera a resposta
            response_text = generate_response_with_llm(final_prompt, llm_model)

            # 4. Exibe a resposta com efeito de digitação
            simulate_typing_effect(response_text)
            
    # 5. Adiciona a resposta do assistente ao histórico
    st.session_state.messages.append({"role": "assistant", "content": response_text})

