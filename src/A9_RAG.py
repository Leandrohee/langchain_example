from typing import Annotated, TypedDict, Sequence, NotRequired, Literal, Any
from pydantic import SecretStr
import os, time
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_chroma import Chroma

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
start_time = time.time()

# --------------------------------------------- MODELS-------------------------------------------- #
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key=SecretStr(api_key) if api_key else None,
    temperature=0,  # For raging
)

# The embedding model has to compatible with the llm
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ------------------------------------- EMBEDDING + CHUNKING ------------------------------------- #

# 1. Get the absolute path of the directory where rag.py lives (the 'src' folder)
SRC_DIR = Path(__file__).resolve().parent

# 2. Go up one level to the project root, then down into 'docs'
PDF_PATH = SRC_DIR.parent / "docs" / "BG-039-02mar2026.pdf"
pdf_path_str = str(PDF_PATH)

if not os.path.exists(pdf_path_str):
    raise FileNotFoundError(f"Pdf not found in: {pdf_path_str}")

# This load the pdf
pdf_loader = PyPDFLoader(pdf_path_str)

# Check if the pdf is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDf: {e}")
    raise

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pages_split = text_splitter.split_documents(pages)

# ---------------------------------------- VECTOR DATABASE --------------------------------------- #

# Persistence directory for the database
db_directory = str(SRC_DIR.parent / "db")
collection_name = "bg_data"

# Create the directory if not exists
if not os.path.exists(db_directory):
    print(f"Created this directory: {db_directory}")
    os.makedirs(db_directory)

# Creating the chroma database using the embedding model
try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=db_directory,
        collection_name=collection_name,
    )
except Exception as error:
    print(f"Error at creating the database: {error}")
    raise

# This is how we are going to retrieve the information from the vector database
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


# --------------------------------------------- TOOLS -------------------------------------------- #
@tool
def retriever_tool(query: str) -> str:
    """ "
    This tool searchs and retrieves the information from the Pdf document.
    The Pdf is in portuguese so the search, the query and the text should be in portuguese.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "Não existe informação relevante nesses documentos referente a pergunta"

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Documento: {i + 1}: \n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)


# ------------------------------------------ AGENT STATE ----------------------------------------- #
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# --------------------------------------------- NODES -------------------------------------------- #
def should_continue(state: AgentState):
    """
    Verify if the last message contain tool calls.
    If the last message does not contain any tool calls it end the agent
    """

    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt = """
Você é um assistente inteligente de IA que responde perguntas relacionadas a um Boletim Geral do Corpo de Bombeiros militar do Distrito Ferderal
Esse boletim geral foi carregado em sua base de dados e você deve basear suas respostas em cima desse documento.
Use as ferramentas de retriever disponíveis para responder as perguntas a respeito do conteudo nesse Boletim geral. Você pode fazer multiplas chamadas se necessário.
Se você queser verificar algumas informações antes de responder a pergunta, você pode fazer isso.
Por favor sempre cite a parte especifica do documento que você usou nas suas respostas.
"""

tools_dict = {
    our_tool.name: our_tool for our_tool in tools
}  # Creating a dictionary of our tools


# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}


# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(
            f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}"
        )

        if not t["name"] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(
            ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
        )

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


# --------------------------------------------- GRAPH -------------------------------------------- #

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm", should_continue, {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()

# --------------------------------------- RUNNING THE AGENT -------------------------------------- #


def running_agent():
    print("\n=== RAG AGENT===")

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [
            HumanMessage(content=user_input)
        ]  # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    running_agent()


# ---------------------------------------- END OF PROGRAM --------------------------------------- #
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️ Total execution time: {elapsed_time:.2f} seconds\n")
