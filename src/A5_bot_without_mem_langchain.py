from dotenv import load_dotenv
from typing import TypedDict, List, NotRequired, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)


class AgentState(TypedDict):
    message: List[HumanMessage]
    tokens_used: NotRequired[int]
    answer: NotRequired[str | list[str | dict[Any, Any]]]


def process_message_node(state: AgentState) -> AgentState:
    """This node responde the user message"""

    response = llm.invoke(state["message"])
    state["tokens_used"] = response.response_metadata["token_usage"]["total_tokens"]
    state["answer"] = response.content
    return state


graph = StateGraph(AgentState)
graph.add_node("process_message", process_message_node)
graph.add_edge(START, "process_message")
graph.add_edge("process_message", END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    response = agent.invoke({"message": [HumanMessage(content=user_input)]})
    print(f"\nAI: {response['answer']}")
    print(f"Tokens used: {response['tokens_used']}")
    user_input = input("Enter: ")
