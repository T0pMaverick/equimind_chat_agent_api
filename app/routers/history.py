"""
History router - Manage chat sessions and history
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Optional
from uuid import UUID

from app.database import get_db
from app.schemas import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionDetail,
    ChatSessionUpdate,
    ChatHistoryListResponse,
    ChatHistoryListItem,
    MessageResponse,
    MessageRole
)
from app.models import ChatSession, Message
from app.middleware.rate_limiter import limiter
from loguru import logger

from app.services.vector_store import vector_store
from app.services.llm_service import llm_service


router = APIRouter(prefix="/api/v1/history", tags=["History"])


@router.get("/sessions", response_model=ChatHistoryListResponse)
@limiter.limit("60/minute")
async def list_sessions(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    include_inactive: bool = False,
    db: Session = Depends(get_db)
):
    """
    List all chat sessions with pagination
    
    - Returns sessions ordered by most recent first
    - Includes message count and last message preview
    - Can filter active/inactive sessions
    """
    # Build query
    query = db.query(ChatSession)
    
    if not include_inactive:
        query = query.filter(ChatSession.is_active == True)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    offset = (page - 1) * page_size
    sessions = query.order_by(desc(ChatSession.updated_at)).offset(offset).limit(page_size).all()
    
    # Build response with message counts and previews
    session_items = []
    for session in sessions:
        # Get message count
        message_count = db.query(func.count(Message.id)).filter(
            Message.session_id == session.id
        ).scalar()
        
        # Get last message preview
        last_message = db.query(Message).filter(
            Message.session_id == session.id
        ).order_by(desc(Message.timestamp)).first()
        
        last_message_preview = None
        if last_message:
            preview_text = last_message.content[:100]
            if len(last_message.content) > 100:
                preview_text += "..."
            last_message_preview = preview_text
        
        session_items.append(
            ChatHistoryListItem(
                id=session.id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=message_count,
                last_message_preview=last_message_preview
            )
        )
    
    return ChatHistoryListResponse(
        sessions=session_items,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
@limiter.limit("60/minute")
async def get_session(
    request: Request,
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get a specific chat session with all messages
    
    - Returns complete session details
    - Includes all messages in chronological order
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Get message count
    message_count = db.query(func.count(Message.id)).filter(
        Message.session_id == session_id
    ).scalar()
    
    # Get all messages
    messages = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.timestamp.asc()).all()
    
    message_responses = [
        MessageResponse(
            id=msg.id,
            session_id=msg.session_id,
            role=MessageRole(msg.role),
            content=msg.content,
            timestamp=msg.timestamp,
            tokens_used=msg.tokens_used,
            metadata=msg.metadata
        )
        for msg in messages
    ]
    
    return ChatSessionDetail(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        is_active=session.is_active,
        metadata=session.metadata,
        message_count=message_count,
        messages=message_responses
    )


@router.post("/sessions", response_model=ChatSessionResponse)
@limiter.limit("30/minute")
async def create_session(
    request: Request,
    session_data: ChatSessionCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new chat session
    
    - Creates an empty session with a title
    - Returns session details
    """
    session = ChatSession(
        title=session_data.title,
        session_metadata=session_data.metadata or {}
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    logger.info(f"Created new chat session: {session.id}")
    
    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        is_active=session.is_active,
        metadata=session.metadata,
        message_count=0
    )


@router.patch("/sessions/{session_id}", response_model=ChatSessionResponse)
@limiter.limit("30/minute")
async def update_session(
    request: Request,
    session_id: UUID,
    update_data: ChatSessionUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a chat session
    
    - Can update title and active status
    - Returns updated session details
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Update fields if provided
    if update_data.title is not None:
        session.title = update_data.title
    
    if update_data.is_active is not None:
        session.is_active = update_data.is_active
    
    db.commit()
    db.refresh(session)
    
    # Get message count
    message_count = db.query(func.count(Message.id)).filter(
        Message.session_id == session_id
    ).scalar()
    
    logger.info(f"Updated chat session: {session.id}")
    
    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        is_active=session.is_active,
        metadata=session.metadata,
        message_count=message_count
    )


@router.delete("/sessions/{session_id}")
@limiter.limit("30/minute")
async def delete_session(
    request: Request,
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a chat session
    
    - Permanently deletes session and all associated messages
    - Cannot be undone
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    db.delete(session)
    db.commit()
    
    logger.info(f"Deleted chat session: {session_id}")
    
    return {"message": "Session deleted successfully", "session_id": str(session_id)}


@router.post("/sessions/{session_id}/archive")
@limiter.limit("30/minute")
async def archive_session(
    request: Request,
    session_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Archive a chat session (mark as inactive)
    
    - Sets is_active to False
    - Session remains in database but hidden from active list
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    session.is_active = False
    db.commit()
    
    logger.info(f"Archived chat session: {session_id}")
    
    return {"message": "Session archived successfully", "session_id": str(session_id)}