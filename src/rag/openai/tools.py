from langchain_core.tools import tool
from embedding import connection_database
from models import llm


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searchs and retrieves the information from the Pdf document.
    The Pdf is in portuguese, so the search, the query and the text should be in portuguese.
    """

    retriever = connection_database()
    docs = retriever.invoke(query)

    if not docs:
        return "Não existe informação relevante nesses documentos referente a pergunta"

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Documento: {i + 1}: \n{doc.page_content}")

    return "\n\n".join(results)


TOOLS = [retriever_tool]

llm_with_tools = llm.bind_tools(TOOLS)
