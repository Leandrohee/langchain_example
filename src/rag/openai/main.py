from langchain_core.messages import HumanMessage, AIMessage
from embedding import connection_database
from graph import rag_agent


def running_agent():

    while True:
        print("\n==================== 🧑 USER ====================")
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke(
            {"messages": messages}, config={"configurable": {"thread_id": "user1"}}
        )

        # print("\n==================== 📨 MESSAGES ====================\n")
        # print(result["messages"])

        print("\n==================== 🤖 ANSWER ====================")
        last_result: AIMessage = result["messages"][-1]
        print(f"\nAI: {last_result.content}")


if __name__ == "__main__":
    running_agent()
