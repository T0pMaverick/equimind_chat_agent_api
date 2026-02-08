"""
Vector store service using ChromaDB with sentence-transformers embeddings
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from app.config import settings
from loguru import logger
import os


class VectorStoreService:
    """Service for managing vector database operations"""
    
    def __init__(self):
        """Initialize Chroma client and embedding model"""
        # Initialize embedding model
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        self.embedding_model = SentenceTransformer(settings.embedding_model_name)
        logger.info(f"Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # Create persist directory if it doesn't exist
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=settings.chroma_collection_name
            )
            logger.info(f"Loaded existing collection: {settings.chroma_collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=settings.chroma_collection_name,
                metadata={"description": "Stock market knowledge base"}
            )
            logger.info(f"Created new collection: {settings.chroma_collection_name}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """Add documents to the vector store"""
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Generate embeddings
        embeddings = self.get_embeddings(documents)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully added {len(documents)} documents")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of search results with content, metadata, and scores
        """
        if top_k is None:
            top_k = settings.vector_search_top_k
        
        # Generate query embedding
        query_embedding = self.get_embeddings([query])[0]
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                score = 1 - results['distances'][0][i]  # Convert distance to similarity
                
                # Only include results above threshold
                if score >= settings.vector_search_score_threshold:
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': float(score),
                        'id': results['ids'][0][i]
                    })
        
        logger.info(f"Found {len(formatted_results)} relevant documents for query")
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        count = self.collection.count()
        return {
            "name": settings.chroma_collection_name,
            "count": count,
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension()
        }
    
    def delete_collection(self):
        """Delete the collection (use with caution)"""
        self.client.delete_collection(name=settings.chroma_collection_name)
        logger.warning(f"Deleted collection: {settings.chroma_collection_name}")


# Global instance
vector_store = VectorStoreService()