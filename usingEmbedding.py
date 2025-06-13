import os
import streamlit as st
import google.generativeai as genai
import numpy as np
import faiss
from dotenv import load_dotenv

st.set_page_config(page_title="Chatbot IA com RAG", layout="wide")

# ------------------------------------------------------------------
# Configuração da API do Google GenAI
# ------------------------------------------------------------------
load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    st.error("Variável de ambiente GENAI_API_KEY não definida. Defina sua API key.")
    st.stop()
genai.configure(api_key=api_key)

# ------------------------------------------------------------------
# Modelos
# ------------------------------------------------------------------
EMBED_MODEL = 'text-embedding-004'
LLM_MODEL = genai.GenerativeModel('gemini-1.5-flash')

# ------------------------------------------------------------------
# Carregamento e criação do índice FAISS (cache para performance)
# ------------------------------------------------------------------
@st.cache_resource
def load_faiss_index():
    # Base de documentos (pode externalizar em arquivo JSON ou DB)
    documentos = {
        "doc1": "O Produto A é uma ferramenta de software de última geração para análise de dados. Ele utiliza algoritmos de machine learning para prever tendências de mercado. Custa R$500 por mês e oferece suporte prioritário 24/7.",
        "doc2": "O Produto B é um serviço de armazenamento em nuvem com 1TB de espaço. A segurança é garantida com criptografia de ponta a ponta. O preço é R$50 por mês.",
        "doc3": "Nossa empresa, a 'Soluções Tech', foi fundada em 2010. Somos líderes no mercado de inovação tecnológica e temos escritórios em São Paulo e no Rio de Janeiro."
    }
    textos = list(documentos.values())
    # Gera embeddings
    res = genai.embed_content(model=EMBED_MODEL, content=textos)
    embeddings = np.array(res['embedding'], dtype='float32')
    # Normaliza para calcular similaridade de cosseno
    faiss.normalize_L2(embeddings)
    # Cria índice Inner Product
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index, textos

index, textos_base = load_faiss_index()

# ------------------------------------------------------------------
# Cabeçalho da interface
# ------------------------------------------------------------------
st.title("🤖 Chatbot IA com Streamlit e RAG")

# ------------------------------------------------------------------
# Estado da sessão para armazenar histórico
# ------------------------------------------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------------------
# Função para recuperar múltiplos contextos
# ------------------------------------------------------------------
def recuperar_contextos(query: str, top_k: int = 3) -> list[str]:
    q_res = genai.embed_content(model=EMBED_MODEL, content=query)
    q_vec = np.array(q_res['embedding'], dtype='float32').reshape(1, -1)
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, top_k)
    return [textos_base[i] for i in indices[0] if i != -1]

# ------------------------------------------------------------------
# Input do usuário
# ------------------------------------------------------------------
with st.form(key='input_form', clear_on_submit=True):
    user_input = st.text_input("Faça sua pergunta:")
    submit = st.form_submit_button("Enviar")

# ------------------------------------------------------------------
# Processa a mensagem quando o usuário envia
# ------------------------------------------------------------------
if submit and user_input:
    # Recupera contextos
    contextos = recuperar_contextos(user_input, top_k=3)
    if contextos:
        separator = "\n\n---\n\n"
        contexto_unido = separator.join(contextos)
        prompt = (
            "Com base estritamente no contexto fornecido, responda à pergunta do usuário.\n"
            "Se a resposta não estiver clara no contexto, responda \"Não tenho informações sobre isso no meu conhecimento.\"\n\n"
            f"Contextos:\n{contexto_unido}\n\n"
            f"Pergunta: {user_input}"
        )
    else:
        prompt = user_input
    # Gera resposta
    chat_response = LLM_MODEL.generate_content(prompt).text
    # Adiciona ao histórico
    st.session_state.history.append(("Você", user_input))
    st.session_state.history.append(("Chatbot", chat_response))

# ------------------------------------------------------------------
# Exibe histórico de chat
# ------------------------------------------------------------------
for speaker, msg in st.session_state.history:
    if speaker == "Você":
        st.markdown(f"**Você:** {msg}")
    else:
        st.markdown(f"**Chatbot:** {msg}")
