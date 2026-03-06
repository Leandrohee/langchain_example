from models import AgentState
from tools import llm_with_tools, TOOLS
from typing import Sequence
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from config import logger
import time


def model_call(state: AgentState) -> AgentState:
    """This function calls for the main LLM with the current state"""

    print("\n==================== ℹ️  LLM INFO ====================\n")

    start_time = time.time()

    system_prompt: SystemMessage = SystemMessage(
        content="""
    Você é um assistente inteligente de IA que responde perguntas relacionadas a um Boletim Geral do Corpo de Bombeiros militar do Distrito Ferderal
    Esse boletim geral foi carregado em sua base de dados e você deve basear suas respostas em cima desse documento.
    Use as ferramentas de retriever disponíveis para responder as perguntas a respeito do conteudo nesse Boletim geral. Você pode fazer multiplas chamadas se necessário.
    Se você queser verificar algumas informações antes de responder a pergunta, você pode fazer isso.
    Por favor sempre cite a parte especifica do documento que você usou nas suas respostas.
    """
    )

    messages: Sequence[BaseMessage] = [system_prompt, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    end_time = time.time()

    if response.usage_metadata:
        elapsed_time = end_time - start_time
        total_tokens = response.usage_metadata["total_tokens"]
        logger.info(f"Tokens usados: {total_tokens}")
        logger.info(f"Tempo de execução: {elapsed_time:.2f} segundos")

    return {"messages": [response]}


def retriever_call(state: AgentState) -> AgentState:
    """
    Execute tool calls from the LLM's response.

    This is a custom tool node.
    Instead of using ToolNode(tools=tools) on graph we used it like this.
    """

    print("\n==================== 🔧 TOOLS ====================\n")

    last_message = state["messages"][-1]
    results: Sequence = []
    # Creating a dictionary of our tools
    tools_dict = {our_tool.name: our_tool for our_tool in TOOLS}

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool in last_message.tool_calls:
            # Checking if the tool name exists
            if not tool["name"] in tools_dict:
                logger.error(
                    "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
                )

            else:
                logger.info("ℹ️ Calling for the tool: %s", tool["name"])
                result = tools_dict[tool["name"]].invoke(tool["args"].get("query", ""))

                results.append(
                    ToolMessage(
                        tool_call_id=tool["id"], name=tool["name"], content=str(result)
                    )
                )

                # Printando todos os kwargs encontrados
                print(f"\n{result}")

    else:
        logger.info("ℹ️ None tool was called")

    return {"messages": results}


def should_continue(state: AgentState):
    """
    Verify if the last message has any tool call

    Return:
        - continue_edge: call the retriever_call
        - end_edge: end the graph
    """

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue_edge"
    else:
        return "end_edge"
