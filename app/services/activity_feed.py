"""
Activity feed service for tracking and streaming agent activities
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID
from app.schemas import ActivityFeedItem, ActivityType
from sqlalchemy.orm import Session
from app.models import ActivityLog
import asyncio
from loguru import logger


class ActivityFeedService:
    """Service for managing activity feed"""
    
    def __init__(self):
        """Initialize activity feed service"""
        self.activities: List[ActivityFeedItem] = []
    
    def add_activity(
        self,
        activity_type: ActivityType,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add an activity to the feed"""
        activity = ActivityFeedItem(
            type=activity_type,
            message=message,
            metadata=metadata or {}
        )
        self.activities.append(activity)
        logger.debug(f"Activity added: {activity_type} - {message}")
    
    def get_activities(self) -> List[ActivityFeedItem]:
        """Get all activities in the feed"""
        return self.activities
    
    def clear(self):
        """Clear all activities"""
        self.activities = []
    
    async def save_to_db(
        self,
        db: Session,
        session_id: UUID,
        message_id: Optional[UUID] = None
    ):
        """Save activities to database"""
        try:
            for activity in self.activities:
                log = ActivityLog(
                    session_id=session_id,
                    message_id=message_id,
                    activity_type=activity.type.value,
                    description=activity.message,
                    metadata=activity.metadata
                )
                db.add(log)
            
            db.commit()
            logger.info(f"Saved {len(self.activities)} activities to database")
        except Exception as e:
            logger.error(f"Error saving activities to database: {str(e)}")
            db.rollback()
    
    # Predefined activity messages
    @staticmethod
    def searching_knowledge_base(query: str) -> Dict[str, str]:
        return {
            "type": ActivityType.SEARCH.value,
            "message": f"Searching internal knowledge base for: '{query[:100]}...'"
        }
    
    @staticmethod
    def found_documents(count: int) -> Dict[str, str]:
        return {
            "type": ActivityType.SEARCH.value,
            "message": f"Found {count} relevant documents in knowledge base"
        }
    
    @staticmethod
    def no_documents_found() -> Dict[str, str]:
        return {
            "type": ActivityType.SEARCH.value,
            "message": "No relevant documents found in knowledge base"
        }
    
    @staticmethod
    def analyzing_financial_metrics() -> Dict[str, str]:
        return {
            "type": ActivityType.ANALYSIS.value,
            "message": "Analyzing financial metrics and market data"
        }
    
    @staticmethod
    def building_investment_thesis() -> Dict[str, str]:
        return {
            "type": ActivityType.ANALYSIS.value,
            "message": "Building investment thesis and risk assessment"
        }
    
    @staticmethod
    def evaluating_catalysts() -> Dict[str, str]:
        return {
            "type": ActivityType.ANALYSIS.value,
            "message": "Evaluating catalysts and timeline projections"
        }
    
    @staticmethod
    def performing_valuation() -> Dict[str, str]:
        return {
            "type": ActivityType.ANALYSIS.value,
            "message": "Performing valuation analysis and sensitivity testing"
        }
    
    @staticmethod
    def generating_report() -> Dict[str, str]:
        return {
            "type": ActivityType.GENERATION.value,
            "message": "Generating comprehensive investment report"
        }
    
    @staticmethod
    def generating_response() -> Dict[str, str]:
        return {
            "type": ActivityType.GENERATION.value,
            "message": "Generating response based on analysis"
        }
    
    @staticmethod
    def creating_alert(alert_name: str) -> Dict[str, str]:
        return {
            "type": ActivityType.ALERT_CREATION.value,
            "message": f"Creating alert: {alert_name}"
        }
    
    @staticmethod
    def alert_created_successfully(alert_id: int) -> Dict[str, str]:
        return {
            "type": ActivityType.ALERT_CREATION.value,
            "message": f"Alert created successfully (ID: {alert_id})"
        }
    
    @staticmethod
    def alert_creation_failed(error: str) -> Dict[str, str]:
        return {
            "type": ActivityType.ERROR.value,
            "message": f"Failed to create alert: {error}"
        }
    
    @staticmethod
    def error_occurred(error: str) -> Dict[str, str]:
        return {
            "type": ActivityType.ERROR.value,
            "message": f"Error: {error}"
        }
    
    @staticmethod
    def processing_query() -> Dict[str, str]:
        return {
            "type": ActivityType.SEARCH.value,
            "message": "Processing your query"
        }
    
    @staticmethod
    def extracting_context() -> Dict[str, str]:
        return {
            "type": ActivityType.SEARCH.value,
            "message": "Extracting relevant context from knowledge base"
        }
    
    @staticmethod
    def preparing_analysis() -> Dict[str, str]:
        return {
            "type": ActivityType.ANALYSIS.value,
            "message": "Preparing detailed analysis"
        }


def create_activity_feed() -> ActivityFeedService:
    """Factory function to create a new activity feed"""
    return ActivityFeedService()