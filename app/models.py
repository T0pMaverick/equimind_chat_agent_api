"""
SQLAlchemy database models
"""
from sqlalchemy import Column, String, Boolean, Integer, Text, ForeignKey, DateTime, JSON, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.database import Base


class ChatSession(Base):
    """Chat session model"""
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    session_metadata = Column(JSONB, default={})
    
    # Relationships
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    activity_logs = relationship("ActivityLog", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    """Message model"""
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    message_metadata = Column(JSONB, default={})
    tokens_used = Column(Integer, default=0)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    activity_logs = relationship("ActivityLog", back_populates="message", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="check_message_role"),
    )


class ActivityLog(Base):
    """Activity log model for tracking agent activities"""
    __tablename__ = "activity_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=True)
    activity_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    activity_metadata = Column(JSONB, default={})
    
    # Relationships
    session = relationship("ChatSession", back_populates="activity_logs")
    message = relationship("Message", back_populates="activity_logs")