"""
Alert service for creating and managing alerts via external API
"""
import httpx
from typing import Dict, Any, List
from app.config import settings
from app.schemas import AlertCreate, AlertResponse, PredefinedMetric
from loguru import logger


class AlertService:
    """Service for managing stock alerts via external API"""
    
    def __init__(self):
        """Initialize HTTP client"""
        self.base_url = settings.alert_api_base_url
        self.timeout = httpx.Timeout(30.0, connect=10.0)
        self.user_id = "5"  # Default user_id
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with user_id"""
        return {
            "Content-Type": "application/json",
            "user_id": self.user_id
        }
    
    async def get_predefined_metrics(self) -> List[Dict[str, Any]]:
        """
        Get list of predefined metrics available for alerts
        
        Returns:
            List of predefined metrics
        """
        url = f"{self.base_url}/alerts/predefined-metrics"
        
        logger.info("Fetching predefined metrics")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._get_headers())
                response.raise_for_status()
                
                metrics = response.json()
                logger.info(f"Retrieved {len(metrics)} predefined metrics")
                return metrics
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching metrics: {e.response.status_code}")
            raise Exception(f"Failed to fetch metrics: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request error fetching metrics: {str(e)}")
            raise Exception(f"Failed to connect to alert service: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error fetching metrics: {str(e)}")
            raise
    
    async def create_alert(self, alert_data: AlertCreate) -> Dict[str, Any]:
        """
        Create a new alert via API
        
        Args:
            alert_data: Alert creation data
            
        Returns:
            Created alert response
        """
        url = f"{self.base_url}/alerts"
        
        payload = alert_data.model_dump()
        
        logger.info(f"Creating alert: {alert_data.name}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url, 
                    json=payload, 
                    headers=self._get_headers()
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Alert created successfully: ID {result.get('id')}")
                return result
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error creating alert: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Failed to create alert: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Request error creating alert: {str(e)}")
            raise Exception(f"Failed to connect to alert service: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error creating alert: {str(e)}")
            raise
    
    async def get_alert(self, alert_id: int) -> Dict[str, Any]:
        """Get alert by ID"""
        url = f"{self.base_url}/alerts/{alert_id}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching alert {alert_id}: {str(e)}")
            raise
    
    async def update_alert(self, alert_id: int, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing alert"""
        url = f"{self.base_url}/alerts/{alert_id}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.put(
                    url, 
                    json=update_data, 
                    headers=self._get_headers()
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error updating alert {alert_id}: {str(e)}")
            raise
    
    async def delete_alert(self, alert_id: int) -> bool:
        """Delete an alert"""
        url = f"{self.base_url}/alerts/{alert_id}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(url, headers=self._get_headers())
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Error deleting alert {alert_id}: {str(e)}")
            raise
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """
        Validate and format ticker symbols to CSE format
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            List of validated symbols in full CSE format
        """
        validated = []
        for symbol in symbols:
            # Ensure symbol is in full format (e.g., JKH.N0000)
            if '.' not in symbol:
                # Add default exchange code if missing
                symbol = f"{symbol}.N0000"
            validated.append(symbol.upper())
        return validated


# Global instance
alert_service = AlertService()