from typing import Annotated, TypedDict, Sequence, NotRequired
from pydantic import SecretStr
import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# AGENTE STATE
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tokens_used: NotRequired[int]
    total_tokens: NotRequired[int]


# TOOLS
@tool
def add_tool(a: int, b: int) -> int:
    """This is an addition function that adds 2 numbers together"""

    return a + b


@tool
def subtractor_tool(a: int, b: int) -> int:
    """This is a subtractor function that subtracts 2 numbers"""

    return a - b


@tool
def multiply_tool(a: int, b: int) -> int:
    """This is a multiply function that multiply 2 numbers"""

    return a * b


tools = [add_tool, subtractor_tool, multiply_tool]

# MODEL
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7,
    api_key=SecretStr(api_key) if api_key else None,
).bind_tools(tools)


# NODES
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )

    # Creates a new list containing the system_prompt and the items from state["messages"]
    messages = [system_prompt, *state["messages"]]

    response = llm.invoke(messages)
    state["tokens_used"] = response.response_metadata["token_usage"]["total_tokens"]

    current_total = state.get("total_tokens", 0)
    state["total_tokens"] = current_total + state["tokens_used"]
    state["messages"] = [response]

    print(f"Tokens in this node: {state['tokens_used']}")
    print(f"Tokens total in this node: {state['total_tokens']}")
    return state


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # Verify it the message is the type that suports tool_calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue_edge"

    return "end_edge"


# GRAPH
graph = StateGraph(AgentState)
graph.add_node("model_call_node", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "model_call_node")
graph.add_conditional_edges(
    "model_call_node", should_continue, {"continue_edge": "tool_node", "end_edge": END}
)
graph.add_edge("tool_node", "model_call_node")
app = graph.compile()


# RUNNING THE CODE
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# inputs: AgentState = {"messages": [HumanMessage(content="Add 40 + 12. Add 3 + 4")]}
inputs: AgentState = {
    "messages": [
        HumanMessage(content="Add 40 + 12 and then multiply the result by 3.")
    ],
}
print_stream(app.stream(inputs, stream_mode="values"))
