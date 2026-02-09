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
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service
from app.services.alert_service import alert_service

from app.services.activity_feed import create_activity_feed

from app.middleware.rate_limiter import limiter
from loguru import logger

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Main chat endpoint with RAG, activity feed, and alert creation
    
    - Searches internal knowledge base (and optionally web)
    - Generates AI response using OpenAI
    - Creates alerts if requested
    - Returns activity feed showing agent's steps
    """
    # Create activity feed
    feed = create_activity_feed()
    
    try:
        # Step 1: Get or create session
        feed.add_activity(
            feed.processing_query().get("type"),
            feed.processing_query().get("message")
        )
        
        session_id = chat_request.session_id
        if session_id:
            # Load existing session
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found")
        else:
            # Create new session
            session = ChatSession(
                title=chat_request.message[:100],  # Use first 100 chars as title
                session_metadata={"context_mode": chat_request.context_mode.value}
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = session.id
        
        # Step 2: Search knowledge base (if internal context enabled)
        context = None
        if chat_request.context_mode in [ContextMode.WEB_AND_INTERNAL, ContextMode.INTERNAL_ONLY]:
            feed.add_activity(
                feed.searching_knowledge_base(chat_request.message).get("type"),
                feed.searching_knowledge_base(chat_request.message).get("message")
            )
            
            search_results = vector_store.search(
                query=chat_request.message,
                top_k=5
            )
            
            if search_results:
                feed.add_activity(
                    feed.found_documents(len(search_results)).get("type"),
                    feed.found_documents(len(search_results)).get("message")
                )
                
                # Build context from search results
                context = "\n\n".join([
                    f"**Document {i+1}** (Relevance: {r['score']:.2f}):\n{r['content']}"
                    for i, r in enumerate(search_results)
                ])
            else:
                feed.add_activity(
                    feed.no_documents_found().get("type"),
                    feed.no_documents_found().get("message")
                )
        
        # Step 3: Get chat history
        history_messages = db.query(Message).filter(
            Message.session_id == session_id
        ).order_by(Message.timestamp.asc()).all()
        
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in history_messages[-10:]  # Last 10 messages
        ]
        
        # Step 4: Determine analysis activities based on reasoning mode
        if chat_request.reasoning_mode == ReasoningMode.DEEP:
            feed.add_activity(
                feed.analyzing_financial_metrics().get("type"),
                feed.analyzing_financial_metrics().get("message")
            )
            feed.add_activity(
                feed.building_investment_thesis().get("type"),
                feed.building_investment_thesis().get("message")
            )
            feed.add_activity(
                feed.evaluating_catalysts().get("type"),
                feed.evaluating_catalysts().get("message")
            )
            feed.add_activity(
                feed.performing_valuation().get("type"),
                feed.performing_valuation().get("message")
            )
            feed.add_activity(
                feed.generating_report().get("type"),
                feed.generating_report().get("message")
            )
        else:
            feed.add_activity(
                feed.preparing_analysis().get("type"),
                feed.preparing_analysis().get("message")
            )
            feed.add_activity(
                feed.generating_response().get("type"),
                feed.generating_response().get("message")
            )
        
        # Step 5: Generate LLM response
        llm_response = await llm_service.generate_response(
            user_message=chat_request.message,
            context=context,
            chat_history=chat_history,
            reasoning_mode=chat_request.reasoning_mode
        )
        
        # Step 6: Save user message
        user_message = Message(
            session_id=session_id,
            role=MessageRole.USER.value,
            content=chat_request.message,
            message_metadata={"context_mode": chat_request.context_mode.value}
        )
        db.add(user_message)
        
        # Step 7: Save assistant message
        assistant_message = Message(
            session_id=session_id,
            role=MessageRole.ASSISTANT.value,
            content=llm_response['content'],
            tokens_used=llm_response['usage']['total_tokens'],
            message_metadata=llm_response['usage']
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        
        # Step 8: Handle alert creation if requested
        alert_created = None
        follow_up_question = None
        
        if chat_request.create_alert:
            try:
                feed.add_activity(
                    feed.creating_alert("Stock Alert").get("type"),
                    feed.creating_alert("Stock Alert").get("message")
                )
                
                # STEP 1: Fetch predefined metrics
                logger.info("Fetching predefined metrics for alert creation")
                predefined_metrics = await alert_service.get_predefined_metrics()
                
                # STEP 2: Extract alert info with conversational approach
                alert_extraction = await llm_service.extract_alert_info_conversational(
                    user_message=chat_request.message,
                    chat_history=chat_history,
                    predefined_metrics=predefined_metrics
                )
                
                if alert_extraction.get("ready"):
                    # We have all info - create the alert
                    alert_data = AlertCreate(
                        name=alert_extraction["alert"]["name"],
                        description=alert_extraction["alert"]["description"],
                        status="active",
                        symbols=alert_extraction["alert"]["symbols"],
                        conditions=alert_extraction["alert"]["conditions"]
                    )
                    
                    alert_response = await alert_service.create_alert(alert_data)
                    alert_created = alert_response
                    
                    feed.add_activity(
                        feed.alert_created_successfully(alert_response.get('id', 'N/A')).get("type"),
                        feed.alert_created_successfully(alert_response.get('id', 'N/A')).get("message")
                    )
                    
                    # Modify assistant message to confirm alert creation
                    confirmation = f"\n\n✅ **Alert Created Successfully!**\n- Name: {alert_data.name}\n- Symbol(s): {', '.join(alert_data.symbols)}\n- Condition: {alert_data.conditions[0]['conditions'][0]['metric']} {alert_data.conditions[0]['conditions'][0]['operation']} {alert_data.conditions[0]['conditions'][0]['values'][0]}"
                    assistant_message.content += confirmation
                    
                else:
                    # Need more info - ask follow-up question
                    follow_up_question = alert_extraction.get("question")
                    logger.info(f"Need more info for alert: {follow_up_question}")
                    
                    # Modify assistant message to include follow-up question
                    assistant_message.content = follow_up_question
                    
                    feed.add_activity(
                        "alert_creation",
                        "Requesting additional information for alert"
                    )
                
            except Exception as e:
                logger.error(f"Error creating alert: {str(e)}")
                feed.add_activity(
                    feed.alert_creation_failed(str(e)).get("type"),
                    feed.alert_creation_failed(str(e)).get("message")
                )
                
                # Add error message to response
                error_msg = f"\n\n⚠️ **Alert Creation Failed**: {str(e)}"
                assistant_message.content += error_msg
        
        # Step 9: Save activity feed to database
        await feed.save_to_db(db, session_id, assistant_message.id)
        
        # Step 10: Return response
        return ChatResponse(
            session_id=session_id,
            message=MessageResponse(
                id=assistant_message.id,
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=assistant_message.content,
                timestamp=assistant_message.timestamp,
                tokens_used=assistant_message.tokens_used,
                metadata=assistant_message.message_metadata
            ),
            activity_feed=feed.get_activities(),
            alert_created=alert_created
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        feed.add_activity(
            feed.error_occurred(str(e)).get("type"),
            feed.error_occurred(str(e)).get("message")
        )
        raise HTTPException(status_code=500, detail=str(e))


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
            await asyncio.sleep(0.1)  # Small delay for UI
            
            # Get or create session
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
            
            # Step 2: Search knowledge base
            context = None
            if chat_request.context_mode in [ContextMode.WEB_AND_INTERNAL, ContextMode.INTERNAL_ONLY]:
                yield send_activity("search", f"Searching internal knowledge base for: '{chat_request.message[:50]}...'")
                await asyncio.sleep(0.1)
                
                search_results = vector_store.search(
                    query=chat_request.message,
                    top_k=5
                )
                
                if search_results:
                    yield send_activity("search", f"Found {len(search_results)} relevant documents in knowledge base")
                    context = "\n\n".join([
                        f"**Document {i+1}** (Relevance: {r['score']:.2f}):\n{r['content']}"
                        for i, r in enumerate(search_results)
                    ])
                else:
                    yield send_activity("search", "No relevant documents found in knowledge base")
                
                await asyncio.sleep(0.1)
            
            # Step 3: Get chat history
            history_messages = db.query(Message).filter(
                Message.session_id == session_id
            ).order_by(Message.timestamp.asc()).all()
            
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in history_messages[-10:]
            ]
            
            # Step 4: Analysis activities
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
            
            # Step 5: Generate LLM response
            llm_response = await llm_service.generate_response(
                user_message=chat_request.message,
                context=context,
                chat_history=chat_history,
                reasoning_mode=chat_request.reasoning_mode
            )
            
            # Step 6: Save messages
            user_message = Message(
                session_id=session_id,
                role=MessageRole.USER.value,
                content=chat_request.message,
                message_metadata={"context_mode": chat_request.context_mode.value}
            )
            db.add(user_message)
            
            assistant_message = Message(
                session_id=session_id,
                role=MessageRole.ASSISTANT.value,
                content=llm_response['content'],
                tokens_used=llm_response['usage']['total_tokens'],
                message_metadata=llm_response['usage']
            )
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)
            
            # Step 7: Handle alert if requested
            alert_created = None
            if chat_request.create_alert:
                try:
                    yield send_activity("alert_creation", "Fetching available alert metrics")
                    await asyncio.sleep(0.1)
                    
                    # Fetch predefined metrics
                    predefined_metrics = await alert_service.get_predefined_metrics()
                    
                    yield send_activity("alert_creation", "Analyzing alert requirements")
                    await asyncio.sleep(0.1)
                    
                    # Extract alert info conversationally
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
                        
                        yield send_activity("alert_creation", f"Alert created successfully (ID: {alert_response.get('id', 'N/A')})")
                    else:
                        # Need more info
                        yield send_activity("alert_creation", "Additional information needed for alert")
                
                except Exception as e:
                    logger.error(f"Error creating alert: {str(e)}")
                    yield send_activity("error", f"Failed to create alert: {str(e)}")
                    
            
            # Step 8: Send final response
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
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )