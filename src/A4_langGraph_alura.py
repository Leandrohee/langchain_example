import os, time, asyncio
from dotenv import load_dotenv
from typing import Literal, TypedDict, cast
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END


# VARIABES
start_time = time.time()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# MODEL
model = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.5,
    api_key=SecretStr(api_key) if api_key else None,
)

prompt_beach = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a travel specialist focusing on beaches"),
        ("human", "{query}"),
    ]
)
prompt_mountain = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a travel specialist focusing on mountain"),
        ("human", "{query}"),
    ]
)
prompt_router = ChatPromptTemplate.from_messages(
    [
        ("system", "Responde only with: 'beach' or 'mountain'"),
        ("human", "{query}"),
    ]
)


# RETURN CLASSES
class DestinyReturn(TypedDict):
    destiny: Literal["beach", "mountain"]


class StateReturn(TypedDict):
    query: str
    destiny: DestinyReturn
    answer: str


# CHAINS
chain_router = prompt_router | model.with_structured_output(DestinyReturn)
chain_beach = prompt_router | model | StrOutputParser()
chain_mountain = prompt_router | model | StrOutputParser()


# KNOTS
async def knot_router(state: StateReturn, config: RunnableConfig | None):
    return {"destiny": await chain_router.ainvoke({"query": state["query"]}, config)}


async def knot_beach(state: StateReturn, config: RunnableConfig | None):
    return {"answer": await chain_beach.ainvoke({"query": state["query"]}, config)}


async def knot_mountain(state: StateReturn, config: RunnableConfig | None):
    return {"answer": await chain_mountain.ainvoke({"query": state["query"]}, config)}


def chosing_knot(state: StateReturn) -> Literal["beach", "mountain"]:
    return "beach" if state["destiny"]["destiny"] == "beach" else "mountain"


# LANGGRAPH
grafo = StateGraph(StateReturn)
grafo.add_node("route", knot_router)
grafo.add_node("beach", knot_beach)
grafo.add_node("mountain", knot_mountain)

grafo.add_edge(START, "route")
grafo.add_conditional_edges("route", chosing_knot)
grafo.add_edge("beach", END)
grafo.add_edge("mountain", END)

app = grafo.compile()


async def main():
    initial_state = cast(
        StateReturn, {"query": "I want to visit great beaches in Brazil"}
    )
    answer = await app.ainvoke(initial_state)
    print(answer)


asyncio.run(main())

# PRINT RESULT
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds\n")
