# app.py
import os
import re
import json
import uuid
import time
import base64
import asyncio
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request as StarletteRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------- DB (your schema, unchanged) ----------
from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

class ChatThread(Base):
    __tablename__ = 'chat_threads'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    messages = relationship('ChatMessage', back_populates='thread', cascade='all, delete-orphan')

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, ForeignKey('chat_threads.id'))
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    thread = relationship('ChatThread', back_populates='messages')

engine = create_engine('sqlite:///chat.sqlite3', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base.metadata.create_all(engine)

# ---------- OpenAI Agents SDK (Azure) ----------
from openai import AzureOpenAI  # azure client

# NOTE: The PyPI package is named `openai-agents` (dash), and it is imported as `openai_agents` (underscore).
# Make sure to install:  pip install openai-agents mcp
try:
    from agents import Agent, Runner, gen_trace_id, trace
    from agents.mcp import MCPServer, MCPServerSse
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency 'openai_agents'. Install it via: pip install openai-agents mcp"
    ) from e

# ---------- FastAPI ----------
app = FastAPI(title="Website Exploration API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- DTOs to mirror your API ----------
class ChatThreadCreateRequest(BaseModel):
    name: Optional[str] = None

class ChatMessageRequest(BaseModel):
    thread_id: str
    message: str

class ChatMessageResponse(BaseModel):
    role: str
    content: str

class ChatHistoryResponse(BaseModel):
    thread_id: str
    history: List[Dict[str, Any]]

class ScreenshotResponse(BaseModel):
    screenshot_base64: Optional[str]
    timestamp: Optional[str]
    status: str

# ---------- Helpers ----------
def free_port() -> int:
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def b64_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _json_dump_safe(obj: Any, max_len: int = 6000) -> str:
    try:
        s = json.dumps(obj)
        if len(s) > max_len:
            return s[:max_len] + "...(truncated)"
        return s
    except Exception:
        txt = str(obj)
        return txt[:max_len] + ("...(truncated)" if len(txt) > max_len else "")

# ---------- WebSocket manager for live screenshots ----------
class ScreenshotWebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.last_sent: Dict[str, str] = {}  # thread_id -> last b64

    async def connect(self, thread_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(thread_id, []).append(websocket)

    def disconnect(self, thread_id: str, websocket: WebSocket):
        if thread_id in self.active_connections and websocket in self.active_connections[thread_id]:
            self.active_connections[thread_id].remove(websocket)

    async def broadcast(self, thread_id: str, img_b64: str, timestamp: str):
        if not img_b64:
            return
        # Avoid re-sending the same frame endlessly
        if self.last_sent.get(thread_id) == img_b64:
            return
        self.last_sent[thread_id] = img_b64
        conns = self.active_connections.get(thread_id, [])
        for ws in conns:
            try:
                await ws.send_json({"type": "screenshot", "data": img_b64, "timestamp": timestamp})
            except Exception:
                pass

screenshot_ws_manager = ScreenshotWebSocketManager()

# ---------- Agent system prompt (updated tool names + strict JSON contract) ----------
SYSTEM_PROMPT = """You are an autonomous Test Exploration Agent that drives a real browser via the Playwright MCP server.

# Goals
1) Read the user's intent and perform actions using Playwright MCP tools.
2) After every meaningful step, report progress briefly (one sentence).
3) If credentials or clarification are required, ask explicitly and wait for the user.
4) Keep working until you can produce a final enhanced test case (Preconditions, Steps, Expected Results as Markdown).
5) Preferred action order: browser_navigate -> browser_wait_for -> browser_click -> browser_type -> (optional actions) -> browser_take_screenshot as evidence.

# Available MCP tools (names must match exactly)
- browser_navigate(url: string)
- browser_click(selector: string)
- browser_type(selector: string, text: string)
- browser_wait_for(selector: string, timeout?: number)
- browser_take_screenshot()
- browser_close()

# Response contract
Always respond in STRICT JSON with exactly these keys:
{
  "content": "<your status/progress/final answer in natural language>",
  "requires_user_input": <true|false>
}

Rules:
- If you need credentials or clarification, set "requires_user_input": true and clearly state what you need in "content".
- Otherwise set "requires_user_input": false and continue autonomously.
- Do not include any other keys in your top-level JSON response.
- When you need to act, use tool calls (browser_*). After tool output, summarize progress in "content".
- If a selector fails, consider taking a screenshot (browser_take_screenshot) and ask for guidance.
"""

# ---------- Per-thread state ----------
@dataclass
class ThreadState:
    thread_id: str
    name: Optional[str] = None

    # MCP
    mcp_port: Optional[int] = None
    mcp_proc: Optional[subprocess.Popen] = None
    mcp_server: Optional[MCPServer] = None
    trace_id : Optional[str] = None
    # Agent
    agent: Optional[Agent] = None

    # Screenshots
    latest_screenshot_b64: Optional[str] = None
    screenshot_timestamp: Optional[str] = None
    screenshot_task: Optional[asyncio.Task] = None

    # Loop state
    waiting_for_user_input: bool = False

    async def start_mcp(self, headless: bool = False):
        """Launch one Playwright MCP server per thread and connect via SSE."""
        self.mcp_port = free_port()
        # The official package in examples is invoked as @playwright/mcp; keep this unless your fork differs.
        cmd = ["npx", "@playwright/mcp@latest", "--port", str(self.mcp_port)]

        # Start Node process
        self.mcp_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # Wait a moment for the server to bind
        await asyncio.sleep(1.0)

        base_url = f"http://127.0.0.1:{self.mcp_port}/sse"
        self.mcp_server = await MCPServerSse(
            name="Playwright mcp server",
            params={
                "url": base_url
            },
        ) 
        self.trace_id = gen_trace_id()
        await self.mcp_transport.connect()
        trace(workflow_name=f"Playwright traces for thread {self.thread_id}", trace_id=self.trace_id)
        print(f'🚀 MCP server started on port {self.mcp_port} for thread {self.thread_id}')

    async def stop_mcp(self):
        
        if self.mcp_proc and self.mcp_proc.poll() is None:
            try:
                self.mcp_proc.terminate()
                self.mcp_proc.wait(timeout=5)
            except Exception:
                self.mcp_proc.kill()
        self.mcp_proc = None
        self.mcp_server = None

    async def start_screenshot_loop(self, interval: float = 2.0):
        if self.screenshot_task and not self.screenshot_task.done():
            return

        async def screenshot_worker():
            while True:
                try:
                    token = await self._invoke_screenshot_tool()
                    if not token:
                        await asyncio.sleep(interval)
                        continue

                    if isinstance(token, str) and token.startswith("base64:"):
                        b64 = token.split("base64:", 1)[1]
                    elif isinstance(token, str) and os.path.exists(token):
                        b64 = b64_of_file(token)
                    else:
                        b64 = None

                    if b64:
                        self.latest_screenshot_b64 = b64
                        self.screenshot_timestamp = datetime.utcnow().isoformat()
                        await screenshot_ws_manager.broadcast(
                            self.thread_id, b64, self.screenshot_timestamp
                        )
                except Exception:
                    pass
                await asyncio.sleep(interval)

        self.screenshot_task = asyncio.create_task(screenshot_worker())

    async def _invoke_screenshot_tool(self) -> Optional[str]:
        """Try the canonical Playwright MCP screenshot tool name, with fallback."""
        candidates = [
            ("browser_take_screenshot", {}),
            ("screenshot", {}),  # some forks expose this alias
        ]
        for name, args in candidates:
            try:
                out = await self.call_mcp_tool(name, args)
                data = getattr(out, "output", out)
                # Accept both {path: "..."} and {base64: "..."} shapes
                if isinstance(data, dict):
                    if "base64" in data and isinstance(data["base64"], str):
                        return "base64:" + data["base64"]
                    path = data.get("path") or data.get("file") or data.get("filepath")
                    if path:
                        return path
                elif isinstance(data, str):
                    # Could be a raw path or raw base64; assume path unless explicitly marked
                    return data
            except Exception:
                continue
        return None


# Global thread registry
THREADS_STATE: Dict[str, ThreadState] = {}

# ---------- Agent factory ----------
def make_azure_agent(system_prompt: str) -> Agent:
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    # IMPORTANT: tool adapter registration – the Agent will use MCPToolClient to execute tool calls.
    return Agent(
        client=client,
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # e.g., "gpt-4o"
        system=system_prompt,
        tools=[MCPToolClient],
    )

# ---------- Thread helpers ----------
async def ensure_thread(thread_id: Optional[str], name: Optional[str]) -> str:
    if thread_id and thread_id in THREADS_STATE:
        return thread_id
    tid = thread_id or str(uuid.uuid4())

    # DB: create thread row
    db = SessionLocal()
    try:
        th = ChatThread(id=tid, name=name)
        db.add(th)
        db.commit()
    finally:
        db.close()

    state = ThreadState(thread_id=tid, name=name)
    await state.start_mcp(headless=False)
    state.agent = make_azure_agent(SYSTEM_PROMPT)
    THREADS_STATE[tid] = state
    # await state.start_screenshot_loop()
    return tid

def db_append_message(thread_id: str, role: str, content: str):
    db = SessionLocal()
    try:
        db.add(ChatMessage(thread_id=thread_id, role=role, content=content))
        th = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
        if th:
            th.updated_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()

# ---------- Agentic turn (updated: strict JSON; no heuristics) ----------
def _parse_agent_json(text: str) -> Dict[str, Any]:
    """Parse agent JSON; if invalid, coerce into the contract."""
    text = (text or "").strip()
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON must be an object")
        content = str(data.get("content", "")).strip()
        requires = bool(data.get("requires_user_input", False))
        return {"content": content, "requires_user_input": requires}
    except Exception:
        # Fallback: wrap raw text into the contract
        return {"content": text, "requires_user_input": False}

async def run_agent_turn(state: ThreadState, user_text: str) -> Dict[str, Any]:
    """
    One agent turn with internal tool-use loop:
      - keeps calling tools via MCP until it either needs user input or has an answer
      - model must always return strict JSON with {content, requires_user_input}
    """
    db_append_message(state.thread_id, "user", user_text)
    state.messages += Messages.user(user_text)

    while True:
        resp = await state.agent.run(state.messages)

        # Handle tool calls first (the Agent exposes them as resp.tool_calls)
        if getattr(resp, "tool_calls", None):
            for call in resp.tool_calls:
                tool_name = call.name
                tool_args = call.arguments or {}
                try:
                    result = await state.call_mcp_tool(tool_name, tool_args)
                    pretty = getattr(result, "output", result)
                    state.messages += ToolResult.for_call(call, _json_dump_safe({"result": pretty}))
                    db_append_message(state.thread_id, "assistant", f"[{tool_name}] -> {_json_dump_safe(pretty)}")
                except Exception as e:
                    err = f"Tool {tool_name} failed: {e}"
                    state.messages += ToolResult.for_call(call, err)
                    db_append_message(state.thread_id, "assistant", err)
            # After tools, loop so the model can observe results and decide next
            continue

        # No tool calls: model produced normal text
        text = getattr(resp, "output_text", "") or ""
        data = _parse_agent_json(text)

        state.waiting_for_user_input = bool(data.get("requires_user_input", False))
        content = str(data.get("content", ""))

        db_append_message(state.thread_id, "assistant", content)
        state.messages += Messages.assistant(content)
        return {"role": "assistant", "content": content}

# ---------- Your API (replicated) ----------

@app.post("/chat/thread", response_model=dict)
async def create_chat_thread(request: ChatThreadCreateRequest):
    thread_id = await ensure_thread(None, request.name)
    return {"thread_id": thread_id}

@app.post("/chat/message", response_model=ChatMessageResponse)
async def chat_message(request: ChatMessageRequest):
    if request.thread_id not in THREADS_STATE:
        raise HTTPException(status_code=404, detail="Thread not found")
    state = THREADS_STATE[request.thread_id]

    # If agent was waiting for input, we just pass the user message through and resume
    if state.waiting_for_user_input:
        db_append_message(request.thread_id, "user", request.message)
        state.messages += Messages.user(request.message)
        state.waiting_for_user_input = False
        # fall through to run a turn right away

    result = await run_agent_turn(state, request.message)
    # The schema expects {role, content}
    return ChatMessageResponse(role=result["role"], content=result["content"])

@app.get("/chat/history/{thread_id}", response_model=ChatHistoryResponse)
async def get_chat_history(thread_id: str):
    db = SessionLocal()
    try:
        messages = db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).order_by(ChatMessage.timestamp).all()
        history = [
            {"id": m.id, "role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
            for m in messages
        ]
        return ChatHistoryResponse(thread_id=thread_id, history=history)
    finally:
        db.close()

# --- Add a stop endpoint for cleaning up a thread (optional but handy) ---
@app.post("/chat/thread/{thread_id}/stop")
async def stop_thread(thread_id: str):
    state = THREADS_STATE.pop(thread_id, None)
    if not state:
        raise HTTPException(404, "Thread not found")
    if state.screenshot_task:
        state.screenshot_task.cancel()
    await state.stop_mcp()
    # Also remove from DB if you want a "hard delete"
    return {"status": "stopped"}

# ---------- Compatibility endpoints for your React frontend ----------

@app.get("/api/threads")
async def api_get_threads():
    db = SessionLocal()
    try:
        threads = db.query(ChatThread).all()
        result = []
        for thread in threads:
            if thread.messages:
                created_at = min(m.timestamp for m in thread.messages).isoformat()
                updated_at = max(m.timestamp for m in thread.messages).isoformat()
            else:
                created_at = updated_at = thread.created_at.isoformat()
            result.append({
                "id": thread.id,
                "name": thread.name or f"Exploration_{thread.id[:8]}",
                "created_at": created_at,
                "updated_at": updated_at
            })
        return {"success": True, "threads": result}
    finally:
        db.close()

@app.post("/api/threads")
async def api_create_thread():
    thread_id = await ensure_thread(None, None)
    return {"success": True, "thread_id": thread_id}

@app.get("/api/threads/{thread_id}/messages")
async def api_get_thread_messages(thread_id: str):
    db = SessionLocal()
    try:
        messages = db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).order_by(ChatMessage.timestamp).all()
        result = [
            {"id": m.id, "role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()} for m in messages
        ]
        return {"success": True, "messages": result}
    finally:
        db.close()

@app.delete("/api/threads/{thread_id}")
async def api_delete_thread(thread_id: str):
    # stop per-thread MCP & loops
    state = THREADS_STATE.pop(thread_id, None)
    if state:
        if state.screenshot_task:
            state.screenshot_task.cancel()
        await state.stop_mcp()

    # delete from DB
    db = SessionLocal()
    try:
        thread = db.query(ChatThread).filter(ChatThread.id == thread_id).first()
        if thread:
            db.delete(thread)
            db.commit()
            return {"success": True}
        return {"success": False, "error": "Thread not found"}
    finally:
        db.close()

@app.post("/api/message")
async def api_post_message(data: dict):
    thread_id = data.get("thread_id")
    message = data.get("message")
    if not thread_id:
        thread_id = await ensure_thread(None, None)
    if thread_id not in THREADS_STATE:
        raise HTTPException(404, "Thread not found")
    res = await run_agent_turn(THREADS_STATE[thread_id], message)
    return {"success": True, "response": res["content"], "thread_id": thread_id}

@app.post("/api/message/stream")
async def api_post_message_stream(request: StarletteRequest):
    body = await request.json()
    thread_id = body.get("thread_id")
    message = body.get("message")
    if not thread_id:
        thread_id = await ensure_thread(None, None)
    if thread_id not in THREADS_STATE:
        raise HTTPException(404, "Thread not found")

    async def event_stream():
        # save user message first
        db_append_message(thread_id, "user", message)
        state = THREADS_STATE[thread_id]
        state.messages += Messages.user(message)

        # Run one agent step
        resp = await state.agent.run(state.messages)

        async def handle_tool_phase():
            nonlocal resp
            while getattr(resp, "tool_calls", None):
                chunks = []
                for call in resp.tool_calls:
                    tool_name = call.name
                    tool_args = call.arguments or {}
                    try:
                        result = await state.call_mcp_tool(tool_name, tool_args)
                        pretty = getattr(result, "output", result)
                        state.messages += ToolResult.for_call(call, _json_dump_safe({"result": pretty}))
                        db_append_message(thread_id, "assistant", f"[{tool_name}] -> {_json_dump_safe(pretty)}")
                        chunks.append(f"[{tool_name}] -> {_json_dump_safe(prety := pretty)}")
                    except Exception as e:
                        err = f"Tool {tool_name} failed: {e}"
                        state.messages += ToolResult.for_call(call, err)
                        db_append_message(thread_id, "assistant", err)
                        chunks.append(err)
                # stream tool results quickly
                for c in chunks:
                    yield f"data: {{\"content\": \"{c.replace('\\', '\\\\').replace('\"', '\\\"')}\"}}\n\n"
                # ask for next model step
                resp = await state.agent.run(state.messages)

        # handle tools (if any)
        async for _ in handle_tool_phase():
            pass

        # now final/next text (expecting strict JSON)
        text = getattr(resp, "output_text", "") or ""
        data = _parse_agent_json(text)

        state.waiting_for_user_input = bool(data.get("requires_user_input", False))
        content = str(data.get("content", ""))

        db_append_message(thread_id, "assistant", content)
        state.messages += Messages.assistant(content)

        # stream assistant text in chunks of ~50 chars
        for i in range(0, len(content), 50):
            chunk = content[i:i+50]
            yield f"data: {{\"content\": \"{chunk.replace('\\', '\\\\').replace('\"', '\\\"')}\"}}\n\n"

        yield f"data: {{\"complete\": true}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------- Screenshot endpoints ----------

@app.get("/screenshot", response_model=ScreenshotResponse)
async def get_latest_screenshot_global():
    # kept for compatibility with your earlier shape — returns the newest among threads
    latest: Optional[Dict[str, Any]] = None
    for tid, state in THREADS_STATE.items():
        if state.latest_screenshot_b64:
            if not latest or (state.screenshot_timestamp and state.screenshot_timestamp > latest["ts"]):
                latest = {"b64": state.latest_screenshot_b64, "ts": state.screenshot_timestamp}
    if not latest:
        return ScreenshotResponse(screenshot_base64=None, timestamp=None, status="no_screenshot_available")
    return ScreenshotResponse(screenshot_base64=latest["b64"], timestamp=latest["ts"], status="available")

@app.get("/screenshot/{thread_id}", response_model=ScreenshotResponse)
async def get_latest_screenshot_thread(thread_id: str):
    state = THREADS_STATE.get(thread_id)
    if not state:
        raise HTTPException(404, "Thread not found")
    if not state.latest_screenshot_b64:
        return ScreenshotResponse(screenshot_base64=None, timestamp=None, status="no_screenshot_available")
    return ScreenshotResponse(screenshot_base64=state.latest_screenshot_b64, timestamp=state.screenshot_timestamp, status="available")

# ---------- Health ----------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "explorer_initialized": True,
        "agent_ready": True,
        "active_tasks": 0,
        "latest_screenshot_available": any(s.latest_screenshot_b64 for s in THREADS_STATE.values()),
        "threads": list(THREADS_STATE.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }

# ---------- WebSocket for per-thread live screenshots ----------
@app.websocket("/ws/screenshots/{thread_id}")
async def websocket_screenshots(websocket: WebSocket, thread_id: str):
    if thread_id not in THREADS_STATE:
        await websocket.close(code=1008)
        return
    await screenshot_ws_manager.connect(thread_id, websocket)
    try:
        while True:
            await asyncio.sleep(15)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        screenshot_ws_manager.disconnect(thread_id, websocket)
    except Exception:
        screenshot_ws_manager.disconnect(thread_id, websocket)

# ---------- Startup / Shutdown ----------
@app.on_event("startup")
async def startup_event():
    # no global agent to init; threads are created on-demand via /chat/thread or /api/threads
    pass

@app.on_event("shutdown")
async def shutdown_event():
    for tid, state in list(THREADS_STATE.items()):
        if state.screenshot_task:
            state.screenshot_task.cancel()
        try:
            await state.stop_mcp()
        except Exception:
            pass

# ---------- Entrypoint ----------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Website Exploration API (MCP + Azure Agents)...")
    print("📋 Useful endpoints:")
    print("  POST /chat/thread")
    print("  POST /chat/message")
    print("  GET  /chat/history/{thread_id}")
    print("  GET  /screenshot/{thread_id}")
    print("  WS   /ws/screenshots/{thread_id}")
    print("  GET  /health")
    print("🌐 http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
