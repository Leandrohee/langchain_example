from typing import Annotated, TypedDict, Sequence, NotRequired, Literal, Any
from pydantic import SecretStr
import os
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
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------- GLOBAL VARIABLE --------------------------------------- #
document_content = ""


# STATE AGENT
class AgenteState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# --------------------------------------------- TOOLS -------------------------------------------- #
@tool
def update_tool(content: str) -> str:
    """Updates the document with the provided content"""

    global document_content
    document_content = content
    return f"The document has been updated successfully! The current content is: \n{document_content}"


@tool
def save_tool(filename: str) -> str:
    """
    Save the current document into a text file and finish the process

    Args:
        filename: Name for the file
    """

    global document_content

    folder_name = "docs"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    filepath = os.path.join(folder_name, filename)

    try:
        with open(filepath, "w") as file:
            file.write(document_content)
        print(f"\n Document has been saved to: {filename}")
        return f"Document has been saved to: {filename}"

    except Exception as e:
        return f"Error saving the document: {str(e)}"


tolls = [update_tool, save_tool]

# --------------------------------------------- MODEL -------------------------------------------- #
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key=SecretStr(api_key) if api_key else None,
    temperature=0.7,
).bind_tools(tolls)


# --------------------------------------------- NODES -------------------------------------------- #
def agent(state: AgenteState) -> AgenteState:
    system_prompt: SystemMessage = SystemMessage(
        content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

    - If the user wants to update or modify content, use the 'update_tool' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save_tool' tool.
    - Make sure to always show the current document state after modifications.

    The current document content is: {document_content}
    """
    )

    print(f"======================= AGENT =======================")

    if not state["messages"]:
        user_input = input(
            "\nI'm ready to help you update a document. What would you like me to create? "
        )
        print(f"🧑 USER: {user_input}")
        user_message: HumanMessage = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"🧑 USER: {user_input}")
        user_message: HumanMessage = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = llm.invoke(all_messages)

    # If the AI is not using any tool that is a content
    if response.content:
        print(f"\n🤖AI: {response.content}")

    # If the Ai is using a tool there is not content
    if isinstance(response, AIMessage) and response.tool_calls:
        print(f"🤖AI: (USING TOOLS: {[tc['name'] for tc in response.tool_calls]})")

    print(f"===================== END AGENT =====================")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgenteState) -> Literal["continue_edge", "end_edge"]:
    """Determine if we should continue or end the conversation"""

    messages = state["messages"]

    if not messages:
        print(f"Should continue: {True}")
        return "continue_edge"

    for message in reversed(messages):
        # Check if this is a tool message returning from save
        message_str = str(message.content)

        if (
            isinstance(message, ToolMessage)
            and "saved" in message_str.lower()
            and "document" in message_str.lower()
        ):
            print(f"Should continue: {False}")
            return "end_edge"

    return "continue_edge"


# --------------------------------------------- GRAPH -------------------------------------------- #
graph = StateGraph(AgenteState)
graph.add_node("agent_node", agent)
graph.add_node("tools_node", ToolNode(tolls))
graph.add_edge(START, "agent_node")
graph.add_edge("agent_node", "tools_node")
graph.add_conditional_edges(
    "tools_node", should_continue, {"continue_edge": "agent_node", "end_edge": END}
)
app = graph.compile()


# -------------------------------------- RUNNING THE PROGRAM ------------------------------------- #
def print_messages(messages):
    """Function I made to print the messages in a more readable format"""

    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


def run_document_agent():
    print("\n ================== DRAFTER START ==================")

    initial_state: AgenteState = {"messages": []}

    for step in app.stream(initial_state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ================== DRAFTER FINISHED ==================")


if __name__ == "__main__":
    run_document_agent()
