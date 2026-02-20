"""
Chat router - Main chatbot endpoint with RAG and activity feed
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
from fastapi.responses import StreamingResponse
import asyncio
import json

from app.database import get_db
from app.schemas import (
    ChatRequest,
    ChatResponse,
    MessageResponse,
    ActivityFeedItem,
    ContextMode,
    ReasoningMode,
    MessageRole,
    AlertCreate,
    AlertConditionGroup,
    AlertCondition
)
from app.models import ChatSession, Message
from app.services.sql_agent_service import sql_agent_service
from app.services.llm_service import llm_service
from app.services.alert_service import alert_service

from app.services.activity_feed import create_activity_feed

from app.middleware.rate_limiter import limiter
from loguru import logger

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])




@router.get("/sessions/{session_id}/messages")
@limiter.limit("60/minute")
async def get_session_messages(
    request: Request,
    session_id: UUID,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get all messages for a session"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.timestamp.asc()).limit(limit).all()
    
    return {
        "session_id": session_id,
        "messages": [
            MessageResponse(
                id=msg.id,
                session_id=msg.session_id,
                role=MessageRole(msg.role),
                content=msg.content,
                timestamp=msg.timestamp,
                tokens_used=msg.tokens_used,
                metadata=msg.message_metadata
            )
            for msg in messages
        ]
    }
    
@router.post("/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Streaming chat endpoint with real-time activity feed
    
    Returns Server-Sent Events (SSE) stream
    """
    
    async def generate_stream():
        """Generate SSE stream of activities and response"""
        feed = create_activity_feed()
        
        def send_activity(activity_type: str, message: str):
            """Helper to send activity as SSE"""
            activity = {
                "type": "activity",
                "data": {
                    "activity_type": activity_type,
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            return f"data: {json.dumps(activity)}\n\n"
        
        try:
            # Step 1: Processing
            yield send_activity("search", "Processing your query")
            await asyncio.sleep(0.1)
            
            # ── Session handling ──────────────────────────────────────────────
            session_id = chat_request.session_id
            if session_id:
                session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
                if not session:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
                    return
            else:
                session = ChatSession(
                    title=chat_request.message[:100],
                    session_metadata={"context_mode": chat_request.context_mode.value}
                )
                db.add(session)
                db.commit()
                db.refresh(session)
                session_id = session.id

            # ── Get chat history ──────────────────────────────────────────────
            history_messages = db.query(Message).filter(
                Message.session_id == session_id
            ).order_by(Message.timestamp.asc()).all()
            
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in history_messages[-10:]
            ]

            # ── Step 2: Query SQL databases (skip if creating alert) ──────────
            final_content = None
            
            # Check if this is an alert-only request
            is_alert_request = chat_request.create_alert and any(
                keyword in chat_request.message.lower() 
                for keyword in ["alert", "notify", "notification", "tell me when", "let me know when"]
            )
            
            if is_alert_request:
                logger.info("Alert-only request detected - skipping SQL query")
                yield send_activity("alert_creation", "Processing alert request")
                await asyncio.sleep(0.1)
                
            elif chat_request.context_mode in [ContextMode.WEB_AND_INTERNAL, ContextMode.INTERNAL_ONLY]:
                yield send_activity("search", "Connecting to financial databases")
                await asyncio.sleep(0.1)

                yield send_activity("search", f"Generating SQL query for: '{chat_request.message[:60]}...'")
                await asyncio.sleep(0.1)

                try:
                    sql_result = await sql_agent_service.query(
                        user_message=chat_request.message,
                        chat_history=chat_history
                    )

                    databases_used = sql_result.get("databases_used", [])
                    if databases_used:
                        yield send_activity("search", f"Queried: {', '.join(databases_used)}")
                    else:
                        yield send_activity("search", "Database query completed")

                    await asyncio.sleep(0.1)

                    # SQL agent already produced the final answer
                    final_content = sql_result.get("content", "")

                    if final_content and "unable to query" not in final_content.lower():
                        yield send_activity("analysis", "Query returned results — formatting response")
                    else:
                        yield send_activity("analysis", "No results found — will provide general response")
                        final_content = None  # Clear error content

                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"SQL agent error: {str(e)}")
                    yield send_activity("error", f"Database query failed: {str(e)}")
                    final_content = None

            # ── Step 3: LLM fallback (if SQL agent returned nothing) ──────────
            if not final_content and not is_alert_request:
                if chat_request.reasoning_mode == ReasoningMode.DEEP:
                    yield send_activity("analysis", "Analyzing financial metrics and market data")
                    await asyncio.sleep(0.1)
                    yield send_activity("analysis", "Building investment thesis and risk assessment")
                    await asyncio.sleep(0.1)
                    yield send_activity("analysis", "Evaluating catalysts and timeline projections")
                    await asyncio.sleep(0.1)
                    yield send_activity("analysis", "Performing valuation analysis and sensitivity testing")
                    await asyncio.sleep(0.1)
                    yield send_activity("generation", "Generating comprehensive investment report")
                else:
                    yield send_activity("analysis", "Preparing detailed analysis")
                    await asyncio.sleep(0.1)
                    yield send_activity("generation", "Generating response based on analysis")

                await asyncio.sleep(0.1)

                llm_response = await llm_service.generate_response(
                    user_message=chat_request.message,
                    context=None,
                    chat_history=chat_history,
                    reasoning_mode=chat_request.reasoning_mode
                )
                final_content = llm_response["content"]
                tokens_used = llm_response["usage"]["total_tokens"]
                usage_metadata = llm_response["usage"]
                
            elif final_content:
                # SQL agent handled it — apply symbol conversion
                from app.utils.symbol_converter import symbol_converter
                final_content = symbol_converter.convert_symbols_in_text(final_content)
                tokens_used = 0
                usage_metadata = {"source": "sql_agent"}
            else:
                # Alert-only request, content will be set after alert creation
                tokens_used = 0
                usage_metadata = {"source": "alert_only"}

            if not is_alert_request:
                yield send_activity("generation", "Finalizing response")
                await asyncio.sleep(0.1)

            # ── Step 4: Handle alert creation ─────────────────────────────────
            alert_created = None
            if chat_request.create_alert:
                try:
                    yield send_activity("alert_creation", "Fetching available alert metrics")
                    await asyncio.sleep(0.1)

                    predefined_metrics = await alert_service.get_predefined_metrics()

                    yield send_activity("alert_creation", "Analyzing alert requirements")
                    await asyncio.sleep(0.1)

                    alert_extraction = await llm_service.extract_alert_info_conversational(
                        user_message=chat_request.message,
                        chat_history=chat_history,
                        predefined_metrics=predefined_metrics
                    )

                    if alert_extraction.get("ready"):
                        yield send_activity("alert_creation", "Creating alert")
                        await asyncio.sleep(0.1)

                        alert_data = AlertCreate(**alert_extraction["alert"])
                        alert_response = await alert_service.create_alert(alert_data)
                        alert_created = alert_response

                        yield send_activity(
                            "alert_creation",
                            f"Alert created successfully (ID: {alert_response.get('id', 'N/A')})"
                        )
                        
                        # ✅ Generate success message
                        alert_dict = alert_data.model_dump()
                        symbols = ", ".join(alert_dict["symbols"])
                        condition = alert_dict["conditions"][0]["conditions"][0]  # ✅ Use alert_dict!
                        metric = condition.get("metric", "value")
                        operation = condition.get("operation", "")
                        value = condition.get("values", [0])[0]
                        
                        alert_success_message = f"""✅ **Alert Created Successfully!**

**Alert Details:**
- **Name:** {alert_data.name}
- **Symbols:** {symbols}
- **Condition:** {metric} {operation} {value}
- **Status:** Active
- **Alert ID:** {alert_response.get('id', 'N/A')}

Your alert is now active and you'll be notified when the condition is met."""

                        # Override content with alert success message
                        final_content = alert_success_message
                        
                    else:
                        # Need more info - ask follow-up question
                        follow_up = alert_extraction.get("question", "Please provide more details.")
                        yield send_activity("alert_creation", "Additional information needed for alert")
                        final_content = follow_up

                except Exception as e:
                    logger.error(f"Error creating alert: {str(e)}")
                    yield send_activity("error", f"Failed to create alert: {str(e)}")
                    
                    if not final_content:
                        final_content = f"I was unable to create the alert: {str(e)}. Please try again with more details."

            # ── Step 5: Save messages ─────────────────────────────────────────
            user_message_obj = Message(
                session_id=session_id,
                role=MessageRole.USER.value,
                content=chat_request.message,
                message_metadata={"context_mode": chat_request.context_mode.value}
            )
            db.add(user_message_obj)

            assistant_message = Message(
                session_id=session_id,
                role=MessageRole.ASSISTANT.value,
                content=final_content or "I'm ready to help. What would you like to know?",
                tokens_used=tokens_used,
                message_metadata=usage_metadata
            )
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)

            # ── Step 6: Send final SSE response ───────────────────────────────
            final_response = {
                "type": "response",
                "data": {
                    "session_id": str(session_id),
                    "message": {
                        "id": str(assistant_message.id),
                        "content": assistant_message.content,
                        "timestamp": assistant_message.timestamp.isoformat(),
                        "tokens_used": assistant_message.tokens_used
                    },
                    "alert_created": alert_created
                }
            }
            yield f"data: {json.dumps(final_response)}\n\n"

            # End of stream
            yield "data: {\"type\": \"done\"}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming chat: {str(e)}")
            error_msg = {
                "type": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_msg)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
    
@router.get("/databases/status")
async def get_database_status():
    """Check status of all connected databases"""
    info = sql_agent_service.get_database_info()
    return {
        "databases": info,
        "total": len(info),
        "connected": sum(1 for db in info if db["status"] == "connected")
    }