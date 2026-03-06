import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import SecretStr
from typing import Annotated, Sequence, TypedDict

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# -------------------------------------------- MODELS -------------------------------------------- #
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key=SecretStr(api_key) if api_key else None,
    temperature=0,
)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


# ------------------------------------------ AGENT STATE ----------------------------------------- #
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
