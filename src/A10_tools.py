from typing import Annotated, TypedDict, Sequence, NotRequired
from pydantic import SecretStr
import os, random
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# ---------------------------------------- MODEL AND STATE --------------------------------------- #
llm = ChatOpenAI(model="gpt-4.1-nano", api_key=SecretStr(api_key) if api_key else None)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    total_tokens: NotRequired[int]


# --------------------------------------------- TOOLS -------------------------------------------- #
@tool
def add_tool(a: int, b: int) -> int:
    """This is a tool that adds two numbers toguether"""

    return a + b


@tool
def subtract_tool(a: int, b: int) -> int:
    """This is a tool that subtracts two numbers"""

    return a - b


@tool
def name_mammal_tool(animal: str) -> str:
    """This is a tool that name an animal, specific a mammal. It does not name another type or animal besides a mammal"""

    number = random.randint(0, 11)

    if number <= 3:
        return "mammal_1"
    elif number > 3 and number <= 7:
        return "mammal_2"
    else:
        return "mammal_3"


@tool
def name_bird_tool(animal: str) -> str:
    """This is a tool that name an animal, specific a bird. It does nt name another type of animal besides a bird"""

    number = random.randint(0, 11)

    if number <= 3:
        return "bird_1"
    elif number > 3 and number <= 7:
        return "bird_2"
    else:
        return "bird_3"


tools = [add_tool, subtract_tool, name_mammal_tool, name_bird_tool]

# llm = llm.bind_tools(tools)
llm_with_tools = llm.bind_tools(tools)


# --------------------------------------------- NODES -------------------------------------------- #
def model_call(state: AgentState) -> AgentState:
    system_prompt: SystemMessage = SystemMessage(
        content="""
    You are my AI agent that helps me some specifc tasks. I need you to use the tools that you were load with.
    These tools are for eather make some simple maths tasks or to help me name some animals.
    If you need any information before answer fell free to ask. 
    If the answer escape from the scope of the tools, inform that you cannot answer that because is not your job. 
    """
    )

    trimmer = trim_messages(
        max_tokens=500,
        strategy="last",
        token_counter=llm,
        start_on="human",
        include_system=True,
    )

    messages: Sequence[BaseMessage] = [system_prompt, *state["messages"]]
    messages = trimmer.invoke(messages)
    print(f"trim messages: {messages}")
    response = llm_with_tools.invoke(messages)

    # If thre is any tools calls print it:
    if isinstance(response, AIMessage) and response.tool_calls:
        for tool in response.tool_calls:
            print(
                f"\nℹ️ Info: Calling for the tool: '{tool['name']}', args: {tool['args']}"
            )
    else:
        print(f"\nℹ️ Info: None tool was called")

    total_tokens = state.get("total_tokens", 0)

    if response.usage_metadata:
        total_tokens += response.usage_metadata["total_tokens"]
        print(f"ℹ️ Info: Tokens used: {response.usage_metadata['total_tokens']}")

    return {"messages": [response], "total_tokens": total_tokens}


def should_continue(state: AgentState):
    """
    Verify if the last message contain tool calls.
    If the last message does not contain any tool calls and the user asks to exit or terminate the conversarion it exists.
    """

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue_edge"

    return "end_edge"


# --------------------------------------------- GRAPH -------------------------------------------- #
graph = StateGraph(AgentState)
graph.add_node("model_call_node", model_call)
graph.add_node("should_continue_node", should_continue)
graph.add_node("tool_node", ToolNode(tools=tools))
graph.add_edge(START, "model_call_node")
graph.add_conditional_edges(
    "model_call_node",
    should_continue,
    {
        # edge: node
        "continue_edge": "tool_node",
        "end_edge": END,
    },
)
graph.add_edge("tool_node", "model_call_node")
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# -------------------------------------------- RUNNING ------------------------------------------- #
def running_with_invoke():
    print("\n================= TOOLS AGENT =================")

    while True:
        user_input = input("\n🧑 USER: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Inititiating the user input
        user_message: Sequence[HumanMessage] = [HumanMessage(content=user_input)]

        # Calling the graph via invoke
        result = app.invoke(
            {"messages": user_message},
            config={"configurable": {"thread_id": "user1"}},
        )

        # Getting the result of the graph
        messages: Sequence[BaseMessage] = result["messages"]
        total_tokens = result["total_tokens"]
        last_result = messages[-1].content

        # Calculating the tokens on the AImessages based on all messages
        for item in messages:
            if isinstance(item, AIMessage):
                all_tot_tokens += item.response_metadata["token_usage"]["total_tokens"]

        print("\n-------------------------------------------------")
        # print(f"ℹ️ INFO: Messages: {messages}")
        print(f"ℹ️ INFO: Nº of messages: {len(messages)}")
        print(f"ℹ️ INFO: Total tokens: {total_tokens}")
        print("-------------------------------------------------")
        print(f"\n🤖AI: {last_result}")


if __name__ == "__main__":
    running_with_invoke()
