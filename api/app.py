"""Phase 5 FastAPI HTTP + WebSocket interface to Phase4Chat / CrewAI.

Start with: uvicorn api.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
import time
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from agents.guardrails import mask_fixed_pii
from api.policies import add_policy
from api.secure_logging import AuditLogger, new_trace_id


class AskRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1, max_length=80, pattern=r'^[a-zA-Z0-9_-]+$')
    question: str = Field(min_length=1, max_length=2000)


class AnswerPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer: str
    grounded: bool
    sources: list[str]
    route: str
    ticket_id: str | None = None
    top_similarity: float | None = None


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    result: AnswerPayload


class PolicyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    doc_id: str = Field(min_length=6, max_length=28)
    title: str = Field(min_length=5, max_length=120)
    content: str = Field(min_length=40, max_length=2000)


class PolicyResponse(BaseModel):
    trace_id: str
    doc_id: str
    indexed_strategies: list[str]


def create_app(*, chat: Any = None, logger: AuditLogger | None = None,
               policy_writer: Callable[[dict], dict] = add_policy) -> FastAPI:
    # Do not load heavy dependencies or embedding model until first request.
    application = FastAPI(title='Ola Support Agent — Offline Capstone', version='0.5.0')
    audit = logger if logger is not None else AuditLogger()
    cached_chat = chat

    def get_chat():
        nonlocal cached_chat
        if cached_chat is None:
            from agents.session_phase4 import Phase4Chat
            cached_chat = Phase4Chat()
        return cached_chat

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # A validation error occurs before route handlers run, so log it here.
        # Do not log the raw body or exception details (potential PII).
        audit.record(trace_id=new_trace_id(), endpoint=request.url.path,
                     status='rejected', started=time.perf_counter(),
                     error_code='SCHEMA_VALIDATION_FAILED')
        return JSONResponse(status_code=422, content={'detail': 'Invalid request schema'})

    @application.get('/health')
    def health():
        return {'status': 'ok', 'mode': 'mock_llm'}

    @application.post('/ask', response_model=AskResponse)
    def ask(request: AskRequest):
        started = time.perf_counter()
        trace = new_trace_id()
        status = 'success'
        code = ''
        try:
            answer = get_chat().chat(request.session_id, request.question)
            # Phase4Chat performs input screening and output evidence checks.
            result = AnswerPayload.model_validate(answer.model_dump())
            return AskResponse(trace_id=trace, session_id=request.session_id, result=result)
        except ValueError:
            status, code = 'rejected', 'INPUT_OR_OUTPUT_GUARDRAIL'
            raise HTTPException(status_code=400, detail='Request rejected by validation or guardrails')
        except Exception:
            status, code = 'error', 'PROCESSING_FAILED'
            raise HTTPException(status_code=500, detail='Agent processing failed; check local prerequisites')
        finally:
            audit.record(trace_id=trace, endpoint='/ask', status=status, started=started,
                         request_text=request.question, session_id=request.session_id,
                         error_code=code)

    @application.post('/add-document', response_model=PolicyResponse, status_code=201)
    def add_document(request: PolicyRequest):
        started = time.perf_counter()
        trace = new_trace_id()
        status, code = 'success', ''
        try:
            saved = policy_writer(request.model_dump())
            return PolicyResponse(trace_id=trace, **saved)
        except FileExistsError:
            status, code = 'rejected', 'DUPLICATE_DOCUMENT'
            raise HTTPException(status_code=409, detail='Policy ID already exists')
        except ValueError:
            status, code = 'rejected', 'INVALID_DOCUMENT'
            raise HTTPException(status_code=400, detail='Invalid policy document')
        except Exception:
            status, code = 'error', 'INDEX_OR_WRITE_FAILED'
            raise HTTPException(status_code=500, detail='Policy indexing failed; check local model and index')
        finally:
            # Never log policy body (even when it contains PII or secrets).
            audit.record(trace_id=trace, endpoint='/add-document', status=status,
                         started=started, error_code=code)

    @application.websocket('/ws/chat')
    async def websocket_chat(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                # Each message: {"session_id":"session-A", "question":"..."}
                payload = await websocket.receive_json()
                started = time.perf_counter()
                trace = new_trace_id()
                session_id = ''
                question = ''
                status, code = 'success', ''
                try:
                    request = AskRequest.model_validate(payload)
                    session_id, question = request.session_id, request.question
                    # CrewAI is synchronous. Run in a worker to avoid blocking the event loop.
                    from starlette.concurrency import run_in_threadpool
                    answer = await run_in_threadpool(get_chat().chat, session_id, question)
                    result = AnswerPayload.model_validate(answer.model_dump())
                    response = AskResponse(trace_id=trace, session_id=session_id, result=result)
                    await websocket.send_json(response.model_dump())
                except ValueError:
                    status, code = 'rejected', 'GUARDRAIL_OR_VALIDATION'
                    await websocket.send_json({'trace_id': trace, 'error': 'Request rejected'})
                except Exception:
                    status, code = 'error', 'PROCESSING_FAILED'
                    await websocket.send_json({'trace_id': trace, 'error': 'Could not process message'})
                finally:
                    audit.record(trace_id=trace, endpoint='/ws/chat', status=status,
                                 started=started, request_text=question, session_id=session_id,
                                 error_code=code)
        except WebSocketDisconnect:
            # A disconnected client must not stop the server or affect other clients.
            return

    return application


app = create_app()
