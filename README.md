# StarDust Backend

AI-powered conversational backend built with FastAPI, LangGraph, Gemini, and Tavily Search.

StarDust provides an agentic chat experience capable of maintaining conversation history, performing web searches, and offering coding guidance through an integrated LeetCode coach tool.

---

## Features

- Streaming AI responses
- Multi-turn conversations
- Persistent chat memory
- Web Search via Tavily
- LeetCode AI Coach
- Gemini 2.5 Flash integration
- LangGraph-based agent workflow
- FastAPI backend
- Server-Sent Events (SSE) support

---

## Architecture

User Query
↓

FastAPI Endpoint

↓

LangGraph Agent

↓

Gemini 2.5 Flash

↓

Tool Calling

├── Tavily Search

└── LeetCode AI Coach

↓

Streaming Response

↓

Frontend


---

## Tech Stack

| Component | Technology |
|----------|------------|
| Backend | FastAPI |
| LLM | Gemini 2.5 Flash |
| Agent Framework | LangGraph |
| Search | Tavily API |
| Memory | LangGraph MemorySaver |
| Environment | python-dotenv |
| Streaming | SSE |

---

## Installation

Clone repository

```bash
git clone https://github.com/yourusername/StarDust_Backend.git

cd StarDust_Backend
```


Install dependencies


```bash
pip install -r requirements.txt
```


---

## Environment Variables

Create a `.env` file.


```env
GOOGLE_API_KEY=YOUR_KEY

TAVILY_API_KEY=YOUR_KEY
```



---

## Running the Server


```bash
uvicorn app:app --reload
```



Server runs at


```text
http://localhost:8000
```



---

## API Endpoints


### Health Check


```http
GET /
```



Response


```json
{
"status":"Backend running"
}
```



---


### Chat Streaming


```http
GET /chat_stream/{message}
```



Optional


```text
checkpoint_id
```



Example


```text
/chat_stream/Explain Binary Search
```



Returns streamed responses.


---

## Built-in Tools


### Tavily Search


Allows the assistant to search the web.


---

### LeetCode AI Coach


Generates:


- Step-by-step approach
- Optimized solution
- Complexity analysis
- Edge cases
- Key insights



Example


```text
Two Sum Problem
```



Output includes:


- Explanation
- Code
- Complexity
- Tips


---

## Deployment


Compatible with:


- Render
- Railway
- Fly.io
- Docker


A Render deployment configuration is included.


```text
render.yaml
```


---

## Future Improvements


- RAG Integration
- Document Upload Support
- Voice Conversations
- Authentication
- Multiple LLM Providers
- Tool Marketplace


---

## License


MIT License
