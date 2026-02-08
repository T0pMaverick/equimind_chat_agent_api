"""
Script to populate vector database from financial knowledge base JSON file

Usage:
    python scripts/populate_vectordb.py --knowledge-base data/enhanced_financial_rag_sentences.json --clear
"""
import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from loguru import logger


def load_knowledge_base(file_path: str) -> List[Dict[str, Any]]:
    """Load knowledge base from JSON file"""
    logger.info(f"Loading knowledge base from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge base file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} items from knowledge base")
    return data


def prepare_documents(knowledge_base: List[Dict[str, Any]]) -> tuple:
    """
    Prepare documents for vector store from financial data
    
    Expected format:
    {
        "id": "ALLI_2025_September_0",
        "text": "Financial analysis text...",
        "metadata": {
            "company_code": "ALLI",
            "company_name": "Alliance Finance",
            "sector": "DIVERSIFIED FINANCE",
            ...
        }
    }
    
    Returns:
        Tuple of (documents, metadatas, ids)
    """
    documents = []
    metadatas = []
    ids = []
    
    for idx, item in enumerate(knowledge_base):
        # Use the 'text' field as the main document content
        text = item.get('text', '')
        
        if not text:
            logger.warning(f"Skipping item {idx}: empty text field")
            continue
        
        documents.append(text)
        
        # Extract metadata
        item_metadata = item.get('metadata', {})
        
        # Create ChromaDB metadata (must be flat dictionary with simple types)
        metadata = {
            "source": "financial_knowledge_base",
            "index": idx,
            "timestamp": datetime.utcnow().isoformat(),
            "original_id": item.get('id', f"doc_{idx}"),
            
            # Company information
            "company_code": str(item_metadata.get('company_code', '')),
            "company_name": str(item_metadata.get('company_name', '')),
            "sector": str(item_metadata.get('sector', '')),
            "reporting_period": str(item_metadata.get('reporting_period', '')),
            
            # Financial metrics (convert to strings for ChromaDB)
            "revenue_3m": str(item_metadata.get('revenue_3m', '')),
            "profit_3m": str(item_metadata.get('profit_3m', '')),
            "return_on_equity": str(item_metadata.get('return_on_equity', '')),
            "return_on_assets": str(item_metadata.get('return_on_assets', '')),
            "total_assets_bn": str(item_metadata.get('total_assets_bn', '')),
            "dividend_yield": str(item_metadata.get('dividend_yield', '')),
            "earnings_per_share": str(item_metadata.get('earnings_per_share', '')),
            "record_id": str(item_metadata.get('record_id', '')),
        }
        
        metadatas.append(metadata)
        
        # Use the original ID from the data
        doc_id = item.get('id', f"doc_{idx:06d}")
        ids.append(doc_id)
    
    logger.info(f"Prepared {len(documents)} documents for ingestion")
    return documents, metadatas, ids


def populate_vectordb(
    knowledge_base_path: str,
    batch_size: int = 100,
    clear_existing: bool = False
):
    """
    Populate vector database with financial knowledge base
    
    Args:
        knowledge_base_path: Path to knowledge base JSON file
        batch_size: Number of documents to process in each batch
        clear_existing: Whether to clear existing collection
    """
    try:
        # Import vector_store here to avoid scope issues
        from app.services.vector_store import vector_store
        
        # Clear existing collection if requested
        if clear_existing:
            logger.warning("Clearing existing collection")
            vector_store.delete_collection()
            
            # Reinitialize by reimporting the module
            import importlib
            import app.services.vector_store as vs_module
            importlib.reload(vs_module)
            from app.services.vector_store import vector_store
        
        # Load knowledge base
        knowledge_base = load_knowledge_base(knowledge_base_path)
        
        # Prepare documents
        documents, metadatas, ids = prepare_documents(knowledge_base)
        
        # Add documents in batches
        total_docs = len(documents)
        logger.info(f"Starting to add {total_docs} documents in batches of {batch_size}")
        
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            batch_docs = documents[i:batch_end]
            batch_metadata = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]
            
            batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}: documents {i+1} to {batch_end}")
            
            vector_store.add_documents(
                documents=batch_docs,
                metadatas=batch_metadata,
                ids=batch_ids
            )
        
        # Verify
        info = vector_store.get_collection_info()
        logger.info("\n" + "=" * 80)
        logger.info("✅ Vector database populated successfully!")
        logger.info("=" * 80)
        logger.info(f"Collection Name: {info['name']}")
        logger.info(f"Total Documents: {info['count']}")
        logger.info(f"Embedding Dimension: {info['embedding_dimension']}")
        logger.info("=" * 80)
        
        # Test search
        logger.info("\n🔍 Testing search functionality...")
        test_queries = [
            "Alliance Finance profit growth",
            "banking sector performance",
            "dividend yield companies"
        ]
        
        for query in test_queries:
            results = vector_store.search(query, top_k=3)
            logger.info(f"\nQuery: '{query}'")
            logger.info(f"Found {len(results)} results")
            for i, result in enumerate(results, 1):
                company = result['metadata'].get('company_name', 'Unknown')
                score = result['score']
                preview = result['content'][:100].replace('\n', ' ')
                logger.info(f"  {i}. [{company}] Score={score:.3f} | {preview}...")
    
    except Exception as e:
        logger.error(f"❌ Error populating vector database: {str(e)}")
        raise


def load_ticker_mapping(ticker_file_path: str):
    """
    Load ticker mapping and add as a special document
    """
    from app.services.vector_store import vector_store
    
    logger.info(f"\n📊 Loading ticker mapping from: {ticker_file_path}")
    
    if not os.path.exists(ticker_file_path):
        logger.warning(f"⚠️  Ticker file not found: {ticker_file_path}")
        return
    
    with open(ticker_file_path, 'r', encoding='utf-8') as f:
        tickers = json.load(f)
    
    # Create a searchable document for tickers
    ticker_text = "Colombo Stock Exchange (CSE) Ticker Symbols and Company Names:\n\n"
    for ticker_info in tickers:
        ticker_text += f"{ticker_info['ticker']}: {ticker_info['company_name']}\n"
    
    vector_store.add_documents(
        documents=[ticker_text],
        metadatas=[{
            "source": "ticker_mapping",
            "type": "reference",
            "count": str(len(tickers)),
            "company_code": "REFERENCE",
            "company_name": "CSE Ticker Directory",
            "sector": "REFERENCE",
            "reporting_period": "",
            "revenue_3m": "",
            "profit_3m": "",
            "return_on_equity": "",
            "return_on_assets": "",
            "total_assets_bn": "",
            "dividend_yield": "",
            "earnings_per_share": "",
            "record_id": "",
            "timestamp": datetime.utcnow().isoformat(),
            "original_id": "ticker_mapping_001"
        }],
        ids=["ticker_mapping_001"]
    )
    
    logger.info(f"✅ Added ticker mapping: {len(tickers)} companies")


def analyze_knowledge_base(knowledge_base_path: str):
    """
    Analyze and display statistics about the knowledge base
    """
    data = load_knowledge_base(knowledge_base_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 KNOWLEDGE BASE ANALYSIS")
    logger.info("=" * 80)
    
    # Count companies
    companies = set()
    sectors = set()
    periods = set()
    
    for item in data:
        metadata = item.get('metadata', {})
        companies.add(metadata.get('company_name', 'Unknown'))
        sectors.add(metadata.get('sector', 'Unknown'))
        periods.add(metadata.get('reporting_period', 'Unknown'))
    
    logger.info(f"Total Records: {len(data)}")
    logger.info(f"Unique Companies: {len(companies)}")
    logger.info(f"Unique Sectors: {len(sectors)}")
    logger.info(f"Reporting Periods: {len(periods)}")
    logger.info("\nSectors:")
    for sector in sorted(sectors):
        count = sum(1 for item in data if item.get('metadata', {}).get('sector') == sector)
        logger.info(f"  - {sector}: {count} records")
    logger.info("=" * 80 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Populate vector database with financial knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Populate with new data (clears existing)
    python scripts/populate_vectordb.py --knowledge-base data/enhanced_financial_rag_sentences.json --clear
    
    # Add more data (keeps existing)
    python scripts/populate_vectordb.py --knowledge-base data/new_data.json
    
    # Analyze data without populating
    python scripts/populate_vectordb.py --knowledge-base data/enhanced_financial_rag_sentences.json --analyze-only
        """
    )
    parser.add_argument(
        "--knowledge-base",
        type=str,
        required=True,
        help="Path to knowledge base JSON file"
    )
    parser.add_argument(
        "--ticker-file",
        type=str,
        default="ticker_mapping.json",
        help="Path to ticker mapping JSON file (default: enhanced_financial_rag_sentences.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collection before populating"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze the knowledge base without populating"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 VECTOR DATABASE POPULATION SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Knowledge base: {args.knowledge_base}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Clear existing: {args.clear}")
    logger.info(f"Embedding model: {settings.embedding_model_name}")
    logger.info("=" * 80 + "\n")
    
    # Analyze knowledge base
    analyze_knowledge_base(args.knowledge_base)
    
    if args.analyze_only:
        logger.info("Analysis complete. Skipping population (--analyze-only flag set)")
        return
    
    # Populate vector database
    populate_vectordb(
        knowledge_base_path=args.knowledge_base,
        batch_size=args.batch_size,
        clear_existing=args.clear
    )
    
    # Load ticker mapping if available
    if os.path.exists(args.ticker_file):
        load_ticker_mapping(args.ticker_file)
    else:
        logger.warning(f"⚠️  Ticker file not found: {args.ticker_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ VECTOR DATABASE POPULATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\n💡 Next steps:")
    logger.info("  1. Start the API: uvicorn app.main:app --reload")
    logger.info("  2. Test search: Try querying for company names or financial metrics")
    logger.info("  3. Check logs for any warnings or errors")
    logger.info("\n")


if __name__ == "__main__":
    main()