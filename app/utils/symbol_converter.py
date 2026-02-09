"""
Utility to convert short stock symbols to full CSE format
"""
import json
import os
import re
from typing import Dict, Optional
from loguru import logger
from functools import lru_cache


class SymbolConverter:
    """Convert short symbols to full CSE format (e.g., INDO -> INDO.N0000)"""
    
    def __init__(self, ticker_file_path: str = "ticker_mapping.json"):
        """Initialize with ticker mapping file"""
        self.ticker_file_path = ticker_file_path
        self.symbol_map: Dict[str, str] = {}
        self._load_ticker_mapping()
    
    def _load_ticker_mapping(self):
        """Load ticker mapping from JSON file"""
        try:
            if not os.path.exists(self.ticker_file_path):
                logger.warning(f"Ticker mapping file not found: {self.ticker_file_path}")
                return
            
            with open(self.ticker_file_path, 'r', encoding='utf-8') as f:
                tickers = json.load(f)
            
            # Create mapping: short symbol -> full symbol
            for ticker_info in tickers:
                full_symbol = ticker_info.get('ticker', '')
                if '.' in full_symbol:
                    short_symbol = full_symbol.split('.')[0].upper()
                    self.symbol_map[short_symbol] = full_symbol
            
            logger.info(f"Loaded {len(self.symbol_map)} ticker symbols")
        
        except Exception as e:
            logger.error(f"Error loading ticker mapping: {str(e)}")
    
    def convert_symbols_in_text(self, text: str) -> str:
        """
        Convert all short stock symbols to full CSE format in text
        
        Args:
            text: Input text with potential short symbols
            
        Returns:
            Text with symbols converted to full format
        """
        if not self.symbol_map:
            return text
        
        # Pattern to match stock symbols (2-4 uppercase letters)
        # Must be surrounded by word boundaries or specific punctuation
        pattern = r'\b([A-Z]{2,4})\b'
        
        def replace_symbol(match):
            short_symbol = match.group(1)
            
            # Skip common words that might match the pattern
            skip_words = {
                'USD', 'LKR', 'EUR', 'GBP', 'JPY', 'CNY',  # Currencies
                'CEO', 'CFO', 'COO', 'CTO', 'CSE', 'IPO',  # Titles/Acronyms
                'GDP', 'EPS', 'ROE', 'ROA', 'P/E', 'EBIT',  # Financial terms
                'Q1', 'Q2', 'Q3', 'Q4', 'FY', 'YTD',  # Time periods
                'API', 'RAG', 'LLM', 'AI', 'ML', 'IT',  # Tech terms
                'AND', 'OR', 'NOT', 'IF', 'THE', 'FOR'  # Common words
            }
            
            if short_symbol in skip_words:
                return short_symbol
            
            # If it's in our ticker map, convert it
            full_symbol = self.symbol_map.get(short_symbol)
            if full_symbol:
                logger.debug(f"Converting {short_symbol} -> {full_symbol}")
                return full_symbol
            
            return short_symbol
        
        # Replace symbols in text
        converted_text = re.sub(pattern, replace_symbol, text)
        
        return converted_text
    
    def get_full_symbol(self, short_symbol: str) -> Optional[str]:
        """
        Get full symbol for a short symbol
        
        Args:
            short_symbol: Short symbol (e.g., "INDO")
            
        Returns:
            Full symbol (e.g., "INDO.N0000") or None
        """
        return self.symbol_map.get(short_symbol.upper())


# Global instance
symbol_converter = SymbolConverter()