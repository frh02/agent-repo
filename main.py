from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"


from langchain_core.messages import AIMessage

def generation_node(state: Sequence[BaseMessage]):
    result = generate_chain.invoke({"messages": state})
    return [AIMessage(content=result["text"])]  # Ensure this is a list of BaseMessages


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res["text"])]  # If res is a dict

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = HumanMessage(content="""
        Please build and improve this blog post for Medium:

        @LangChainAI's new Tool Calling feature is underrated.

        After a long wait, it's finally here—making it much easier to implement agents across models with function calling.

        I made a video covering their latest blog post.
        """)
    response = graph.invoke(inputs)
    print("This is the response",response)