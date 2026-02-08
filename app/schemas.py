"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


# ============= Enums =============

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContextMode(str, Enum):
    WEB_AND_INTERNAL = "web_internal"
    INTERNAL_ONLY = "internal_only"


class ReasoningMode(str, Enum):
    QUICK = "quick"
    DEEP = "deep"


class ActivityType(str, Enum):
    SEARCH = "search"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    ALERT_CREATION = "alert_creation"
    ERROR = "error"


# ============= Message Schemas =============

class MessageBase(BaseModel):
    role: MessageRole
    content: str


class MessageCreate(MessageBase):
    session_id: Optional[UUID] = None
    metadata: Optional[Dict[str, Any]] = {}


class MessageResponse(MessageBase):
    id: UUID
    session_id: UUID
    timestamp: datetime
    tokens_used: int
    metadata: Dict[str, Any]
    
    class Config:
        from_attributes = True


# ============= Chat Session Schemas =============

class ChatSessionBase(BaseModel):
    title: str


class ChatSessionCreate(ChatSessionBase):
    metadata: Optional[Dict[str, Any]] = {}


class ChatSessionUpdate(BaseModel):
    title: Optional[str] = None
    is_active: Optional[bool] = None


class ChatSessionResponse(ChatSessionBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    is_active: bool
    metadata: Dict[str, Any]
    message_count: Optional[int] = 0
    
    class Config:
        from_attributes = True


class ChatSessionDetail(ChatSessionResponse):
    messages: List[MessageResponse] = []


# ============= Chat Request/Response Schemas =============

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[UUID] = None
    context_mode: ContextMode = ContextMode.WEB_AND_INTERNAL
    reasoning_mode: ReasoningMode = ReasoningMode.QUICK
    create_alert: bool = False
    
    @validator('message')
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()


class ActivityFeedItem(BaseModel):
    type: ActivityType
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = {}


class ChatResponse(BaseModel):
    session_id: UUID
    message: MessageResponse
    activity_feed: List[ActivityFeedItem]
    alert_created: Optional[Dict[str, Any]] = None


# ============= Alert Schemas =============

class AlertCondition(BaseModel):
    metric: str
    operation: str  # <, <=, >, >=, =, BETWEEN, IN
    values: List[float]
    negation: bool = False
    
    @validator('operation')
    def validate_operation(cls, v):
        valid_ops = ['<', '<=', '>', '>=', '=', 'BETWEEN', 'IN']
        if v not in valid_ops:
            raise ValueError(f'Operation must be one of {valid_ops}')
        return v
    
    @validator('values')
    def validate_values(cls, v, values):
        operation = values.get('operation')
        if operation == 'BETWEEN' and len(v) != 2:
            raise ValueError('BETWEEN operation requires exactly 2 values')
        elif operation in ['<', '<=', '>', '>=', '='] and len(v) != 1:
            raise ValueError(f'{operation} operation requires exactly 1 value')
        return v


class AlertConditionGroup(BaseModel):
    operator: str = Field(..., pattern="^(AND|OR)$")
    conditions: List[AlertCondition]


class AlertCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str
    status: str = "active"
    symbols: List[str]
    conditions: List[AlertConditionGroup]
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError('At least one symbol is required')
        return v


class AlertResponse(BaseModel):
    id: int
    name: str
    description: str
    status: str
    symbols: List[str]
    created_at: datetime
    updated_at: datetime
    conditions: List[Dict[str, Any]]


class PredefinedMetric(BaseModel):
    id: int
    metric: str
    data_type: str
    description: str
    data_key: str
    data_class: str


# ============= History Schemas =============

class ChatHistoryListItem(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_preview: Optional[str] = None
    
    class Config:
        from_attributes = True


class ChatHistoryListResponse(BaseModel):
    sessions: List[ChatHistoryListItem]
    total: int
    page: int
    page_size: int


# ============= Vector Search Schemas =============

class VectorSearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float


# ============= Ticker Mapping Schema =============

class TickerInfo(BaseModel):
    ticker: str
    company_name: str