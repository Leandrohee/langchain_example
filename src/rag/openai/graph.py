from langgraph.graph import START, StateGraph, END
from models import AgentState
from nodes import model_call, retriever_call, should_continue
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(AgentState)
graph.add_node("model_call_node", model_call)
graph.add_node("retriever_call_node", retriever_call)
graph.add_edge(START, "model_call_node")
graph.add_conditional_edges(
    "model_call_node",
    should_continue,
    {
        # Edge: node
        "continue_edge": "retriever_call_node",
        "end_edge": END,
    },
)
graph.add_edge("retriever_call_node", "model_call_node")
memory = MemorySaver()
rag_agent = graph.compile(checkpointer=memory)
