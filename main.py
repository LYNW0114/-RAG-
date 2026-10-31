"""FastAPI 问答服务：用户问答统一走这里，知识库上传仍由 Streamlit 负责。"""
import json
import re
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from file_history_store import get_history
from rag import RagService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RagService()
    yield


app = FastAPI(title="私有知识库智能问答 API", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(default=None, description="会话 ID，不传则自动生成")


class Source(BaseModel):
    file: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str


def resolve_session_id(session_id: str | None) -> str:
    if not session_id:
        return uuid.uuid4().hex
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(status_code=400, detail="session_id 只能包含字母、数字、下划线或短横线，长度 1-64")
    return session_id


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs")


@app.get("/health")
def health(request: Request):
    try:
        indexed_chunks = len(request.app.state.rag.vector_service.vector_store.get(include=[])["ids"])
    except Exception:
        indexed_chunks = -1
    return {"status": "ok", "indexed_chunks": indexed_chunks}


@app.get("/api/documents")
def list_documents(request: Request):
    data = request.app.state.rag.vector_service.vector_store.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for metadata in data.get("metadatas") or []:
        source = (metadata or {}).get("source", "未知来源")
        counts[source] = counts.get(source, 0) + 1
    return {
        "total_chunks": len(data.get("ids") or []),
        "documents": [{"file": source, "chunks": chunks} for source, chunks in sorted(counts.items())],
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    session_id = resolve_session_id(payload.session_id)
    try:
        result = await run_in_threadpool(request.app.state.rag.answer, payload.question, session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答失败：{exc}") from exc
    return ChatResponse(answer=result["answer"], sources=result["sources"], session_id=session_id)


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    session_id = resolve_session_id(payload.session_id)
    rag = request.app.state.rag

    async def event_stream():
        yield sse({"type": "start", "session_id": session_id})
        try:
            async for event in rag.astream_answer(payload.question, session_id):
                yield sse(event)
        except Exception as exc:
            yield sse({"type": "error", "message": f"问答失败：{exc}"})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/sessions/{session_id}")
def reset_session(session_id: str):
    session_id = resolve_session_id(session_id)
    get_history(session_id).clear()
    return {"session_id": session_id, "cleared": True}
