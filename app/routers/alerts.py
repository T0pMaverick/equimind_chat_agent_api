"""
Alerts router - Proxy endpoints for alert management
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List

from app.schemas import AlertCreate, AlertResponse, PredefinedMetric
from app.services import alert_service
from app.middleware.rate_limiter import limiter
from loguru import logger

router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])


@router.post("/", response_model=AlertResponse)
@limiter.limit("10/minute")
async def create_alert(
    request: Request,
    alert_data: AlertCreate
):
    """
    Create a new stock alert
    
    - Validates alert parameters
    - Forwards request to alert service API
    - Returns created alert details
    """
    try:
        # Validate and format symbols
        alert_data.symbols = alert_service.validate_symbols(alert_data.symbols)
        
        result = await alert_service.create_alert(alert_data)
        return result
    
    except Exception as e:
        logger.error(f"Error creating alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predefined-metrics", response_model=List[PredefinedMetric])
@limiter.limit("60/minute")
async def get_predefined_metrics(request: Request):
    """
    Get list of predefined metrics available for alerts
    
    - Returns all available metrics with their data types
    - Used for building alert conditions
    """
    try:
        metrics = await alert_service.get_predefined_metrics()
        return metrics
    
    except Exception as e:
        logger.error(f"Error fetching predefined metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{alert_id}")
@limiter.limit("60/minute")
async def get_alert(
    request: Request,
    alert_id: int
):
    """Get alert by ID"""
    try:
        alert = await alert_service.get_alert(alert_id)
        return alert
    
    except Exception as e:
        logger.error(f"Error fetching alert: {str(e)}")
        raise HTTPException(status_code=404, detail="Alert not found")


@router.put("/{alert_id}")
@limiter.limit("10/minute")
async def update_alert(
    request: Request,
    alert_id: int,
    update_data: dict
):
    """Update an existing alert"""
    try:
        result = await alert_service.update_alert(alert_id, update_data)
        return result
    
    except Exception as e:
        logger.error(f"Error updating alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{alert_id}")
@limiter.limit("10/minute")
async def delete_alert(
    request: Request,
    alert_id: int
):
    """Delete an alert"""
    try:
        await alert_service.delete_alert(alert_id)
        return {"message": "Alert deleted successfully", "alert_id": alert_id}
    
    except Exception as e:
        logger.error(f"Error deleting alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))