import google.generativeai as genai
import os
import numpy as np
import faiss  # Biblioteca para busca de similaridade

# --- 1. CONFIGURAÇÃO INICIAL E MODELO DE EMBEDDING ---
try:
    genai.configure(api_key='AIzaSyAXd4FGz6RxCyFg-phb5P65f0cCcTnm9Lk')
except Exception as e:
    print(f"Erro ao configurar a API: {e}")
    exit()

# Modelos a serem usados
embedding_model = 'text-embedding-004'
llm_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. BASE DE CONHECIMENTO (CONTEÚDO ORIGINAL) ---
# Mantemos os textos originais separados
documentos = {
    "doc1": "O Produto A é uma ferramenta de software de última geração para análise de dados. Ele utiliza algoritmos de machine learning para prever tendências de mercado. Custa R$500 por mês e oferece suporte prioritário 24/7.",
    "doc2": "O Produto B é um serviço de armazenamento em nuvem com 1TB de espaço. A segurança é garantida com criptografia de ponta a ponta. O preço é R$50 por mês.",
    "doc3": "Nossa empresa, a 'Soluções Tech', foi fundada em 2010. Somos líderes no mercado de inovação tecnológica e temos escritórios em São Paulo e no Rio de Janeiro."
}

# --- 3. INDEXAÇÃO: GERAR EMBEDDINGS E CRIAR O ÍNDICE VETORIAL ---
# Esta parte é executada uma vez para preparar a base de conhecimento
def criar_indice_vetorial(documentos, model):
    print("Gerando embeddings para a base de conhecimento...")
    
    # Extrai o conteúdo de texto dos documentos
    textos_dos_documentos = list(documentos.values())
    
    # Gera os embeddings para todos os textos de uma vez
    result = genai.embed_content(model=model, content=textos_dos_documentos)
    
    # Converte os embeddings para um array numpy
    embeddings_np = np.array(result['embedding']).astype('float32')
    
    # Cria um índice FAISS, que é um banco de dados vetorial em memória
    faiss.normalize_L2(embeddings_np)

    d = embeddings_np.shape[1]  # Dimensão dos vetores
    index = faiss.IndexFlatIP(d)   # IP = Inner Product
    index.add(embeddings_np)
    
    print("Índice vetorial criado com sucesso.")
    return index, textos_dos_documentos

# Criando o índice ao iniciar o programa
indice_faiss, textos_originais = criar_indice_vetorial(documentos, embedding_model)


# --- 4. FUNÇÃO DE RECUPERAÇÃO (RETRIEVAL) COM EMBEDDINGS ---
def recuperar_contexto_com_embedding(query, index, model, textos_base, top_k=1):
    # 1. Gera o embedding para a pergunta do usuário
    query_embedding_result = genai.embed_content(model=model, content=query)
    query_vector = np.array(query_embedding_result['embedding']).astype('float32').reshape(1, -1)

    faiss.normalize_L2(query_vector)

    # 2. Busca no índice FAISS pelos vetores mais próximos (mais similares)
    # D: Distâncias, I: Índices dos vetores encontrados
    distancias, indices = index.search(query_vector, top_k)
    
    # 3. Retorna o texto original correspondente ao vetor mais próximo encontrado
    if len(indices[0]) > 0:
        indice_encontrado = indices[0][0]
        return textos_base[indice_encontrado]
    return None

def chatbot_com_ia():
    print("Olá! Eu sou um chatbot com IA e RAG baseado em embeddings.")
    print("Digite 'sair' para terminar a conversa.")

    while True:
        entrada_usuario = input("\n> Você: ")

        if entrada_usuario.lower() == 'sair':
            print("Chatbot IA: Até a próxima!")
            break

        if entrada_usuario:
            # Substituímos a busca por palavra-chave pela busca por embedding
            contexto = recuperar_contexto_com_embedding(entrada_usuario, indice_faiss, embedding_model, textos_originais)
            
            prompt_final = entrada_usuario

            if contexto:
                print("--- [Contexto recuperado por similaridade semântica. Usando RAG] ---")
                prompt_final = (
                    f"Com base estritamente no contexto fornecido, responda à pergunta do usuário.\n"
                    f"Se a resposta não estiver clara no contexto, diga 'Não tenho informações sobre isso no meu conhecimento'.\n\n"
                    f"Contexto: '{contexto}'\n\n"
                    f"Pergunta: '{entrada_usuario}'"
                )

            try:
                response = llm_model.generate_content(prompt_final)
                print(f"Chatbot IA: {response.text}")
            except Exception as e:
                print(f"Chatbot IA: Desculpe, ocorreu um erro: {e}")

if __name__ == "__main__":
    chatbot_com_ia()