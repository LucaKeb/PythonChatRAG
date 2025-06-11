import google.generativeai as genai
import os

# --- 1. BASE DE CONHECIMENTO (NOSSO CONTEÚDO PRIVADO PARA O RAG) ---
base_de_conhecimento = {
    "produto a": {
        "content": "O Produto A é uma ferramenta de software de última geração para análise de dados. Ele utiliza algoritmos de machine learning para prever tendências de mercado. Custa R$500 por mês e oferece suporte prioritário 24/7.",
        "keywords": ["produto a", "preço do a", "software", "análise de dados"]
    },
    "produto b": {
        "content": "O Produto B é um serviço de armazenamento em nuvem com 1TB de espaço. A segurança é garantida com criptografia de ponta a ponta. O preço é R$50 por mês.",
        "keywords": ["produto b", "preço do b", "nuvem", "armazenamento", "segurança"]
    },
    "empresa": {
        "content": "Nossa empresa, a 'Soluções Tech', foi fundada em 2010. Somos líderes no mercado de inovação tecnológica e temos escritórios em São Paulo e no Rio de Janeiro.",
        "keywords": ["empresa", "soluções tech", "sobre nós", "escritórios"]
    }
}

# --- 2. FUNÇÃO DE RECUPERAÇÃO (RETRIEVAL) ---
def recuperar_contexto(query, base):
    query_lower = query.lower()
    for item_key, item_data in base.items():
        for keyword in item_data["keywords"]:
            if keyword in query_lower:
                return item_data["content"]
    return None

def chatbot_com_ia():
    try:
        api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyAXd4FGz6RxCyFg-phb5P65f0cCcTnm9Lk")
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Erro ao configurar a API. Verifique sua chave. Detalhes: {e}")
        print("Você pode obter uma chave de API em: https://aistudio.google.com/app/apikey")
        return
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Olá! Eu sou um chatbot com IA e RAG. Posso responder sobre 'Produto A', 'Produto B' e a 'Empresa'.")
    print("Digite 'sair' para terminar a conversa.")

    while True:
        entrada_usuario = input("\n> Você: ")

        if entrada_usuario.lower() == 'sair':
            print("Chatbot IA: Até a próxima!")
            break

        if entrada_usuario:
            
            contexto = recuperar_contexto(entrada_usuario, base_de_conhecimento)
            
            prompt_final = entrada_usuario

            # Aumentando o prompt final
            if contexto:
                print("--- [Contexto encontrado. Usando RAG] ---")
                
                prompt_final = (
                    f"Com base estritamente no contexto fornecido, responda à seguinte pergunta.\n"
                    f"Se a resposta não estiver no contexto, diga 'Não tenho informações sobre isso no meu conhecimento'.\n\n"
                    f"Contexto: '{contexto}'\n\n"
                    f"Pergunta: '{entrada_usuario}'"
                )

            try:
                
                response = model.generate_content(prompt_final)
                print(f"Chatbot IA: {response.text}")
            except Exception as e:
                print(f"Chatbot IA: Desculpe, ocorreu um erro ao contatar a IA. Detalhes: {e}")

if __name__ == "__main__":
    chatbot_com_ia()
