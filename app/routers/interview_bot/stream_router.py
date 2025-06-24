from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import Dict
import uuid
import logging
import asyncio

from app.services.interview_stream_service import InterviewStreamService
from app.base.models import StreamInterviewConfig

router = APIRouter(prefix="/interview_stream", tags=["Interview Bot - Real-Time"])
logger = logging.getLogger("interview_stream")

# In-memory session manager
active_sessions: Dict[str, InterviewStreamService] = {}

@router.websocket("/ws/{session_id}")
async def interview_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"[InterviewStream] 📞 New WebSocket session: {session_id}")

    stream_handler = InterviewStreamService()
    active_sessions[session_id] = stream_handler

    try:
        # Step 1: Receive config
        config_data = await websocket.receive_json()
        interview_config = StreamInterviewConfig(**config_data)
        session = stream_handler.start_session(interview_config)
        logger.info(f"[InterviewStream] Configured session for {interview_config.candidate_id}")

        # Step 2: Receive and process audio chunks
        while True:
            try:
                chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
            except asyncio.TimeoutError:
                logger.warning(f"[InterviewStream] Timeout waiting for chunk: {session_id}")
                break

            answer = await session.handle_stream_chunk(chunk)

            if answer:
                # Once complete response scored → notify frontend
                await websocket.send_json({
                    "event": "scored_answer",
                    "answer": answer.dict()
                })

            # (Optional) Intermediate transcript (for frontend UX)
            # await websocket.send_json({
            #     "event": "partial_transcript",
            #     "text": "..."  # from stream_handler if partial decode supported
            # })

    except WebSocketDisconnect:
        logger.info(f"[InterviewStream] 🚫 Client disconnected: {session_id}")
    except Exception as e:
        logger.exception(f"[InterviewStream] ❌ Error in session {session_id}: {e}")
        await websocket.send_json({"event": "error", "message": str(e)})
    finally:
        try:
            result = stream_handler.end_session(interview_config.candidate_id)
            if result:
                await websocket.send_json({
                    "event": "complete",
                    "result": result.dict()
                })
        except Exception as e:
            logger.warning(f"[InterviewStream] Finalization failed for {session_id}: {e}")
        await websocket.close()
        active_sessions.pop(session_id, None)
