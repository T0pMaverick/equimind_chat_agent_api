"""
Configuration management for the Stock Analysis RAG Agent
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # PostgreSQL Databases for Text-to-SQL
    db_1_url: str = ""
    db_2_url: str = ""
    db_3_url: str = ""
    db_1_name: str = "Database 1"
    db_2_name: str = "Database 2"
    db_3_name: str = "Database 3"
        
    # Application
    app_name: str = Field(default="Stock Analysis RAG Agent", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="production", env="ENVIRONMENT")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Vector Database (Chroma)
    chroma_persist_directory: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="stock_knowledge_base", env="CHROMA_COLLECTION_NAME")
    
    # Embeddings
    embedding_model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        env="EMBEDDING_MODEL_NAME"
    )
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=20, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=200, env="RATE_LIMIT_PER_HOUR")
    
    # Alert API
    alert_api_base_url: str = Field(..., env="ALERT_API_BASE_URL")
    
    # CORS
    allowed_origins: str = Field(
        default="http://localhost:3000",
        env="ALLOWED_ORIGINS",
        
    )
    
    # Chat Configuration
    max_context_messages: int = Field(default=10, env="MAX_CONTEXT_MESSAGES")
    max_tokens_per_response: int = Field(default=2000, env="MAX_TOKENS_PER_RESPONSE")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Vector Search
    vector_search_top_k: int = Field(default=5, env="VECTOR_SEARCH_TOP_K")
    vector_search_score_threshold: float = Field(default=0.6, env="VECTOR_SEARCH_SCORE_THRESHOLD")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse comma-separated origins into a list"""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()