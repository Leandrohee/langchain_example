import os, time
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

start_time = time.time()

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


modelo = ChatOpenAI(
    # model="gpt-3.5-turbo",    # 5.62 seconds  | 757 tokens
    # model="gpt-5-nano",       # 75.38 seconds | 8013 tokens
    # model="gpt-4.1-nano",     # 3.21 seconds  | 482 tokens
    model="gpt-4.1-mini",  # 4.69 seconds   | 505 tokens
    # model="gpt-5-mini",       # 44.75 seconds | 3010 tokens
    temperature=0.5,
    api_key=SecretStr(api_key) if api_key else None,
)

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios",
        ),
        ("placeholder", "{historico}"),
        ("human", "{query}"),
    ]
)

# If I use StrOutputParser I dont have the metadata to retrieve token consumption
# cadeia = prompt_sugestao | modelo | StrOutputParser()
cadeia = prompt_sugestao | modelo

memoria = {}
sessao = "aula_langchain_alura"


def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]


lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?",
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico",
)

tokens_totais: int = 0

for uma_pergunta in lista_perguntas:
    resposta_objeto = cadeia_com_memoria.invoke(
        {"query": uma_pergunta}, config={"configurable": {"session_id": sessao}}
    )

    # 1. Extraindo os tokens e as repostas (forma moderna)
    tokens = resposta_objeto.usage_metadata
    texto_ia = resposta_objeto.content

    # 2. Convertendo em dicionario
    resposta_dic = resposta_objeto.dict()
    resposta_dic.pop("content", None)
    metadados = resposta_dic

    tokens_totais = tokens_totais + tokens["total_tokens"]

    print(f"---")
    print("Usuário: ", uma_pergunta)
    print("Texto IA: ", texto_ia)
    print("Metadados: ", metadados, "\n")
    print(
        f"Consumo: {tokens['input_tokens']} entrada | {tokens['output_tokens']} saída | Total: {tokens['total_tokens']}"
    )


print(f"Total de tokens em todas as interaçoes: {tokens_totais}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds\n")
