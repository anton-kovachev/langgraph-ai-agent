from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


load_dotenv()  # Load environment variables from .env file

llm = init_chat_model("claude-sonnet-4-20250514", temperature=0)


class MessageClassifier(BaseModel):
    """
    Model for classifying a message as emotional, logical, or neutral.
    """
    message_type: Literal["emotional", "logical", "neutral"] = Field(
        ..., description="Classify the message as emotional, logical, or neutral"
    )


class State(TypedDict):
    """
    State object for passing messages and message type through the graph.
    """
    messages: Annotated[list, add_messages]
    message_type: str | None


def classify_message(state: State):
    """
    Classify the last message in the state as emotional, logical, or neutral.

    Parameters:
    state (State): The current state containing messages.

    Returns:
    dict: Dictionary with the classified message type.
    """
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": (
                "Classify the user's message into one of three categories: emotional, logical, or neutral.\n"
                "- Emotional: Messages that express feelings, opinions, or personal experiences.\n"
                "- Logical: Messages that are fact-based, analytical, or seek information.\n"
                "- Neutral: Messages that are general, non-emotional, and do not seek specific information"
            )
        },
        {"role": "user", "content": last_message.content}
    ])

    return {"message_type": result.message_type}  # type: ignore


def router(state: State):
    """
    Route to the appropriate agent node based on message type.

    Parameters:
    state (State): The current state containing message_type.

    Returns:
    dict: Dictionary with the next node name.
    """
    message_type = state.get("message_type", "logical")

    if message_type == "emotional":
        return {"next_node": "emotional"}
    elif message_type == "neutral":
        return {"next_node": "neutral"}
    else:
        return {"next_node": "logical"}


def emotional_agent(state: State):
    """
    Generate an empathetic response to the user's message.

    Parameters:
    state (State): The current state containing messages.

    Returns:
    dict: Dictionary with assistant's reply message.
    """
    last_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content": (
                "You are an empathetic and understanding assistant. "
                "Your responses should be warm, supportive, and considerate of the user's feelings."
            )
        },
        {"role": "user", "content": last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def neutral_agent(state: State):
    """
    Generate a neutral and objective response to the user's message.

    Parameters:
    state (State): The current state containing messages.

    Returns:
    dict: Dictionary with assistant's reply message.
    """
    last_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a neutral and objective assistant. "
                "Your responses should be unbiased, straightforward, and free from emotional influence."
            )
        },
        {"role": "user", "content": last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def logical_agent(state: State):
    """
    Generate a logical and analytical response to the user's message.

    Parameters:
    state (State): The current state containing messages.

    Returns:
    dict: Dictionary with assistant's reply message.
    """
    last_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a logical and analytical assistant. "
                "Your responses should be clear, concise, and focused on providing factual information or solutions."
            )
        },
        {"role": "user", "content": last_message.content}
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


graph_builder = StateGraph(State)

graph_builder.add_node("classify_message", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("emotional_agent", emotional_agent)
graph_builder.add_node("logical_agent", logical_agent)
graph_builder.add_node("neutral_agent", neutral_agent)

graph_builder.add_edge(START, "classify_message")
graph_builder.add_edge("classify_message", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next_node", None),
    path_map={
        "emotional": "emotional_agent",
        "logical": "logical_agent",
        "neutral": "neutral_agent"
    }
)

graph_builder.add_edge("emotional_agent", END)
graph_builder.add_edge("logical_agent", END)
graph_builder.add_edge("neutral_agent", END)

graph = graph_builder.compile()

def run_chatbot(State, graph):
    while True:
        user_input = input("Please, ask me anything:\n")
        if user_input.lower() in {"quit", "exit"}:
            break

        initial_state = State(
            messages=[{"role": "user", "content": user_input}],
            message_type=None
        )

        final_state = graph.invoke(initial_state)
        print(f"Assistant: {final_state['messages'][-1].content}\n")

if __name__ == "__main__":
    run_chatbot(State, graph)