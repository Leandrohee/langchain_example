from typing import TypedDict, NotRequired, Literal
from langgraph.graph import StateGraph, START, END


# AGENT STATE
class AgentState(TypedDict):
    number1: int
    number2: int
    finalNumner: NotRequired[int]
    operation: Literal["+", "-"]


# NODES
def adder(state: AgentState) -> AgentState:
    """This node sum 2 numbers"""

    state["finalNumner"] = state["number1"] + state["number2"]
    return state


def subtractor(state: AgentState) -> AgentState:
    """This node subtracts 2 numbers"""

    state["finalNumner"] = state["number1"] - state["number2"]
    return state


def decide_next_node(state: AgentState):
    """This node select the next node on the graph"""

    if state["operation"] == "+":
        return "addition_edge"
    elif state["operation"] == "-":
        return "subtraction_edge"
    else:
        raise ValueError(f"Unsupported operation: {state['operation']}")


# GRAPH
graph = StateGraph(AgentState)
graph.add_node("adder_node", adder)
graph.add_node("subtractor_node", subtractor)
graph.add_node(
    "router_node", lambda state: state
)  # passtrough function. Get the variable 'state' and passtrought

graph.add_edge(START, "router_node")
graph.add_conditional_edges(
    "router_node",
    decide_next_node,
    {
        # edge: node
        "addition_edge": "adder_node",
        "subtraction_edge": "subtractor_node",
    },
)
graph.add_edge("adder_node", END)
graph.add_edge("subtractor_node", END)
app = graph.compile()

# RUNNING THE PROGRAM
initial_state: AgentState = {"number1": 10, "number2": 3, "operation": "+"}
response = app.invoke(initial_state)
print(f"The answer is: {response['finalNumner']}")
