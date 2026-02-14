from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

load_dotenv()

# ------------------ MEMORY ------------------
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ------------------ TOOLS ------------------

search_tool = TavilySearchResults(max_results=4)

@tool
def leetcode_ai_coach(problem: str, language: str = "python") -> str:
    """
    Returns structured coaching response for a coding problem.
    """
    return f"""
    Analyze the following coding problem:

    {problem}

    Provide:
    1. Step-by-step approach
    2. Optimized solution in {language}
    3. Time complexity
    4. Space complexity
    5. Edge cases
    6. Key insight
    """

tools = [search_tool, leetcode_ai_coach]


# ------------------ GEMINI MODEL ------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",  # Use the EXACT name from the list
    temperature=0.7,
    streaming=True
)

llm_with_tools = llm.bind_tools(tools=tools)

# ------------------ GRAPH NODES ------------------
async def model_node(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [result]}

async def tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END

async def tool_node(state: State):
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for call in tool_calls:
        for tool in tools:
            if tool.name == call["name"]:
                result = tool.invoke(call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=call["id"],
                        name=call["name"]
                    )
                )

    return {"messages": tool_messages}


# ------------------ GRAPH ------------------
graph = StateGraph(State)
graph.add_node("model", model_node)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("model")
graph.add_conditional_edges("model", tools_router)
graph.add_edge("tool_node", "model")
graph = graph.compile(checkpointer=memory)

# ------------------ FASTAPI ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://your-app-name.netlify.app"  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def serialise_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        content = chunk.content
        # Handle if content is a list
        if isinstance(content, list):
            return "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])
        return content if isinstance(content, str) else ""
    return ""

async def generate_chat_responses(message: str, checkpoint_id: Optional[str]):
    if checkpoint_id is None:
        checkpoint_id = str(uuid4())

    config = {"configurable": {"thread_id": checkpoint_id}}

    yield f"data: {json.dumps({'type': 'checkpoint', 'checkpoint_id': checkpoint_id})}\n\n"

    events = graph.astream_events(
        {"messages": [HumanMessage(content=message)]},
        version="v2",
        config=config
    )

    async for event in events:
        etype = event["event"]

        if etype == "on_chat_model_stream":
            text = serialise_chunk(event["data"]["chunk"])
            if text.strip():
                yield f"data: {json.dumps({'type':'content','content':text})}\n\n"

        elif etype == "on_chat_model_end":
            yield f"data: {json.dumps({'type':'end'})}\n\n"
@app.get("/")
async def root():
    return {"status": "Backend running"}

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id),
        media_type="text/event-stream"
    )
