"""
Text-to-SQL Agent Service using LangChain
Connects to multiple PostgreSQL databases and answers natural language questions
"""
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from typing import List, Dict, Any, Optional
from app.config import settings
from loguru import logger
import asyncio


class SQLAgentService:
    """
    Service that manages multiple PostgreSQL database connections
    and routes user questions to appropriate databases using LangChain SQL Agent
    """

    def __init__(self):
        """Initialize LLM and database connections"""
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0  # Keep at 0 for SQL generation accuracy
        )

        # Load all database connections
        self.databases: Dict[str, SQLDatabase] = {}
        self._load_databases()

        # Create combined agent with all DBs
        self.agent = self._create_multi_db_agent()

        logger.info(f"SQL Agent Service initialized with {len(self.databases)} databases")

    def _get_table_descriptions(self) -> Dict[str, str]:
        """
        Provide natural language descriptions for each table
        This helps the agent choose the right table
        """
        return {
            # ── Database 1: Company Metrices ──
            "companies_financial_data": """
                PRIMARY TABLE for company financial metrics and ratios.
                Contains quarterly financial data including:
                - Revenue, profit, EPS (earnings per share)
                - Trailing EPS, Net Asset Value
                - ROE (return on equity), ROA (return on assets)
                - DIVIDEND YIELD (current dividend yield percentage) ← USE THIS FOR DIVIDEND YIELD QUERIES
                - Total assets, total equity
                - Company sector, reporting_period
                
                CRITICAL: reporting_period format is 'YYYY_MonthName' (e.g., '2025_September', '2024_December')
                NOT 'YYYY-MM' format!
                
                Available periods: 2025_September, 2024_September, 2024_December, etc.
                
                Example queries:
                - September 2025 data: WHERE reporting_period = '2025_September'
                - December 2024 data: WHERE reporting_period = '2024_December'
                - All 2025 data: WHERE reporting_period LIKE '2025_%'
                
                Columns include:
                - company_code (NOT company_name) - use this for company identifier
                - dividend_yield, earnings_per_share, trailing_eps
                - return_on_equity, return_on_assets
                - sector, total_assets_bn, total_equity_bn
                
                Use this table for:
                - Financial ratios (EPS, ROE, ROA, dividend yield)
                - Company fundamentals
                - Quarterly performance metrics
                - "Top companies by [any financial metric]"
            """,
            
            "historical_stock_data": """
                Historical price and volume data (OHLCV).
                Contains: open, high, low, close prices, volume
                Use for: price history, trading volumes, price trends
            """,
            
            "sector_summary": """
                Aggregated sector-level statistics.
                Use for: sector comparisons, industry analysis
            """,
            
            # ── Database 2: Company Financials ──
            "financial_metrics": """
                Additional financial analysis metrics.
                Use for: detailed financial calculations, custom metrics
            """,
            
            "dividend_history": """
                Historical dividend PAYMENT records (past dividends paid).
                Contains: dividend payment dates, amounts, types (cash/scrip)
                
                DO NOT USE for current dividend yield queries.
                Only use for: dividend payment history, past dividend dates
            """,
            
            # ── Database 3: Trading Data ──
             "stock_analysis_all_results": """
            ⭐ PRIMARY TABLE FOR TECHNICAL INDICATORS AND STOCK ANALYSIS ⭐
            
            Contains comprehensive technical analysis data for all stocks:
            
            TECHNICAL INDICATORS:
            - RSI (Relative Strength Index) ← USE THIS FOR RSI QUERIES
            - rsi_divergence (Bearish/Bullish divergence detection)
            - relative_strength (market relative strength)
            - EMA values (ema_20, ema_50, ema_100, ema_200)
            - volume_analysis (momentum indicators)
            
            PRICE DATA:
            - closing_price, prev_close, change_pct
            - turnover, volume, vol_avg_5d, vol_avg_20d
            
            CRITICAL COLUMN NAMES:
            - symbol (stock ticker e.g., 'HNB.N0000', 'COMB.N0000')
            - rsi (RSI value as decimal, e.g., 46.09)
            - date (latest analysis date)
            
            Use this table for:
            - RSI queries ("What is the RSI of HNB?")
            - Technical indicator queries (EMA, volume analysis)
            - Price change analysis
            - Volume momentum analysis
            
            Example queries:
            - RSI: SELECT symbol, rsi, closing_price FROM stock_analysis_all_results WHERE symbol = 'HNB.N0000'
            - Top RSI: SELECT symbol, rsi, closing_price ORDER BY rsi DESC LIMIT 10
        """,
        
        "dividend_history": """
            ⭐ HISTORICAL DIVIDEND PAYMENT RECORDS ⭐
            
            Contains past dividend announcements and ex-dividend dates:
            
            COLUMNS:
            - company_code (symbol format: 'XXXX.N0000')
            - announcement_date (when dividend was announced)
            - rate_of_dividend (dividend amount/rate)
            - xd_date (ex-dividend date - when stock trades without dividend)
            
            Use this table for:
            - Dividend payment history ("When did HNB pay dividends?")
            - Dividend announcement dates
            - Historical dividend rates
            - Ex-dividend date queries
            - Dividend frequency analysis
            
            IMPORTANT: This is for HISTORICAL dividend payments, NOT current yield
            For current dividend yield → use companies_financial_data table
            
            Example queries:
            - Recent dividends: SELECT * FROM dividend_history WHERE company_code = 'HNB.N0000' ORDER BY xd_date DESC LIMIT 5
            - Dividend history for 2024: SELECT * FROM dividend_history WHERE company_code = 'ALLI.N0000' AND xd_date >= '2024-01-01'
            - Total dividends paid: SELECT company_code, SUM(rate_of_dividend) as total_dividends FROM dividend_history GROUP BY company_code
        """,
        
        "pattern_analysis": """
            ⭐ CHART PATTERN ANALYSIS AND TRADING SIGNALS ⭐
            
            Contains AI-generated technical pattern analysis with trading recommendations:
            
            KEY COLUMNS:
            - symbol (stock ticker: 'XXXX.N0000')
            - analysis_date (date of analysis)
            - overall_sentiment (Bullish/Bearish/Neutral)
            - recommended_action (specific trading recommendation text)
            - current_price (price at analysis time)
            - entry_price (suggested entry point)
            - stop_loss (suggested stop loss level)
            - target_price (profit target)
            - raw_payload (JSON with detailed pattern analysis)
            
            The raw_payload JSON contains:
            - chart_patterns (e.g., "Rounded Bottom", "Diamond", "Head and Shoulders")
            - candlestick_patterns (e.g., "Bullish Harami", "Doji", "Engulfing")
            - signal_confidence (reliability score 0-100)
            - pattern rationale (why pattern was detected)
            - detection_reliability (0.0-1.0)
            
            Use this table for:
            - Chart pattern queries ("What patterns does HNB show?")
            - Trading signal requests ("Should I buy COMB?")
            - Sentiment analysis ("Is ALLI bullish or bearish?")
            - Entry/exit price recommendations
            - Pattern-based trading strategies
            - Find stocks with specific patterns
            
            Example queries:
            - Latest analysis: SELECT symbol, overall_sentiment, recommended_action, entry_price, target_price 
                               FROM pattern_analysis WHERE symbol = 'HNB.N0000' ORDER BY analysis_date DESC LIMIT 1
            - Bullish stocks: SELECT symbol, recommended_action, current_price, target_price 
                             FROM pattern_analysis WHERE overall_sentiment = 'Bullish' ORDER BY analysis_date DESC LIMIT 10
            - Pattern details: SELECT symbol, raw_payload FROM pattern_analysis WHERE symbol = 'COMB.N0000' ORDER BY analysis_date DESC LIMIT 1
            - Best opportunities: SELECT symbol, overall_sentiment, (target_price - current_price) as potential_gain 
                                 FROM pattern_analysis WHERE overall_sentiment = 'Bullish' ORDER BY potential_gain DESC LIMIT 10
        """,
        
        "trades": """
            Individual trade transactions.
            Contains: trade_id, symbol, price, volume, timestamp, buyer, seller
            Use for: trade-level analysis, execution prices, order flow analysis
        """,
        
        "orderbook_snapshots": """
            Order book state at different times.
            Contains: symbol, timestamp, bid_levels, ask_levels, spreads
            Use for: market depth, bid/ask analysis, liquidity analysis
        """,
        
        "crossings": """
            Block trades and crossing transactions.
            Contains: symbol, price, volume, crossing_date, participants
            Use for: large institutional trades, off-market deals, block trade analysis
        """,
        
        "day_agg": """
            Daily aggregated trading statistics.
            Contains: symbol, date, total_volume, total_value, trades_count, vwap
            Use for: daily summaries, volume aggregates, daily statistics
        """,
        
        "orderbook_levels": """
            Detailed order book levels.
            Contains: symbol, timestamp, level, bid_price, bid_volume, ask_price, ask_volume
            Use for: market microstructure analysis, depth analysis
        """
        }
    
    def _load_databases(self):
        """Load all configured PostgreSQL databases with custom table info"""
        db_configs = [
            (settings.db_1_url, settings.db_1_name),
            (settings.db_2_url, settings.db_2_name),
            (settings.db_3_url, settings.db_3_name),
        ]
        
        table_descriptions = self._get_table_descriptions()

        for db_url, db_name in db_configs:
            if db_url:
                try:
                    db = SQLDatabase.from_uri(
                        db_url,
                        sample_rows_in_table_info=0,
                        engine_args={
                            "pool_pre_ping": True,
                            "pool_recycle": 300,
                            "pool_timeout": 30,
                            "connect_args": {
                                "connect_timeout": 10,
                                "keepalives": 1,
                                "keepalives_idle": 30,
                                "keepalives_interval": 10,
                                "keepalives_count": 5
                            }
                        },
                        custom_table_info=table_descriptions
                    )
                    self.databases[db_name] = db
                    logger.info(
                        f"✅ Connected to database: {db_name} "
                        f"| Tables: {db.get_usable_table_names()}"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to connect to {db_name}: {str(e)}")

    def _get_all_tools(self) -> List[BaseTool]:
        """Create tools from all databases combined"""
        all_tools = []

        for db_name, db in self.databases.items():
            toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
            tools = toolkit.get_tools()

            # Rename tools to include DB name for clarity
            for tool in tools:
                tool.name = f"{db_name.replace(' ', '_')}_{tool.name}"
                tool.description = f"[{db_name}] {tool.description}"
                all_tools.append(tool)

        return all_tools

    def _build_system_prompt(self) -> str:
        """Build system prompt with database schema information"""
        db_info = ""
        for db_name, db in self.databases.items():
            tables = db.get_usable_table_names()
            db_info += f"\n**{db_name}:**\n"
            db_info += f"Tables: {', '.join(tables)}\n"

        return f"""You are an expert SQL analyst for the Colombo Stock Exchange (CSE) financial data system.

    {db_info}

    **CRITICAL DATE FORMAT RULES:**
    The `reporting_period` column uses format: 'YYYY_MonthName'
    Examples: '2025_September', '2024_December', '2025_March'

    **CRITICAL SYMBOL FORMAT:**
    - Stock symbols in Trading Data DB use format: 'XXXX.N0000' (e.g., 'HNB.N0000', 'COMB.N0000')
    - If user says "HNB", search for 'HNB.N0000' in Trading Data tables
    - If user says "COMB", search for 'COMB.N0000' in Trading Data tables
    - Always add '.N0000' suffix when querying Trading Data tables
    - In Company Metrices DB, symbols may be without suffix (just 'HNB', 'COMB')

    **TABLE SELECTION RULES:**

    1. **RSI queries** (Relative Strength Index):
    → stock_analysis_all_results.rsi

    2. **Technical indicators** (EMA, volume, divergence):
    → stock_analysis_all_results (rsi, ema_20, ema_50, ema_100, ema_200, volume_analysis)

    3. **Chart patterns and trading signals**:
    → pattern_analysis (overall_sentiment, recommended_action, entry_price, stop_loss, target_price)

    4. **Dividend payment history**:
    → dividend_history (rate_of_dividend, xd_date, announcement_date)

    5. **Current dividend yield**:
    → companies_financial_data.dividend_yield

    6. **Financial ratios** (EPS, ROE, ROA):
    → companies_financial_data

    7. **Price history**:
    → historical_stock_data OR stock_analysis_all_results.closing_price

    8. **Trading sentiment/recommendations**:
    → pattern_analysis (overall_sentiment, recommended_action)

    **COMMON QUERY PATTERNS:**

    "What is the RSI of HNB?"
    → SELECT symbol, rsi, date FROM stock_analysis_all_results WHERE symbol = 'HNB.N0000'

    "Show me bullish stocks"
    → SELECT symbol, overall_sentiment, recommended_action, current_price, target_price 
    FROM pattern_analysis WHERE overall_sentiment = 'Bullish' ORDER BY analysis_date DESC LIMIT 10

    "When did ALLI pay dividends?"
    → SELECT company_code, rate_of_dividend, xd_date FROM dividend_history 
    WHERE company_code = 'ALLI.N0000' ORDER BY xd_date DESC

    "What patterns does COMB show?"
    → SELECT symbol, overall_sentiment, recommended_action 
    FROM pattern_analysis WHERE symbol = 'COMB.N0000' ORDER BY analysis_date DESC LIMIT 1

    "Should I buy HNB?"
    → SELECT symbol, overall_sentiment, recommended_action, entry_price, stop_loss, target_price 
    FROM pattern_analysis WHERE symbol = 'HNB.N0000' ORDER BY analysis_date DESC LIMIT 1

    "Current dividend yield of JKH?"
    → SELECT company_code, dividend_yield FROM companies_financial_data 
    WHERE company_code = 'JKH' AND reporting_period = '2025_September'

    "Top 10 by RSI"
    → SELECT symbol, rsi, closing_price FROM stock_analysis_all_results ORDER BY rsi DESC LIMIT 10

    **Rules:**
    - SELECT only (never INSERT, UPDATE, DELETE, DROP)
    - Limit to 50 rows unless specified
    - Handle both symbol formats
    - Use ORDER BY date/analysis_date DESC LIMIT 1 for "latest" or "current"
    - If query fails, try alternative formats
    """
    def _get_query_examples(self) -> str:
        """Provide example queries to guide the agent"""
        return """
    **Example Query Patterns:**

    Q: "Top 10 companies by dividend yield in September 2025"
    A: SELECT company_code, dividend_yield 
    FROM companies_financial_data 
    WHERE reporting_period LIKE '2025-09%' 
    ORDER BY dividend_yield DESC 
    LIMIT 10;

    Q: "What is the EPS of SAMP in Q3 2025?"
    A: SELECT company_code, earnings_per_share, trailing_eps 
    FROM companies_financial_data 
    WHERE company_code = 'SAMP' 
    AND reporting_period LIKE '2025-09%';

    Q: "Companies with ROE above 15%"
    A: SELECT company_code, return_on_equity, sector 
    FROM companies_financial_data 
    WHERE return_on_equity > 15 
    ORDER BY return_on_equity DESC;
    """


    def _create_multi_db_agent(self):
        """Create a single agent with tools from all databases"""
        if not self.databases:
            logger.warning("No databases configured!")
            return None

        # Use first DB as primary for the agent
        primary_db_name = list(self.databases.keys())[0]
        primary_db = self.databases[primary_db_name]

        # Get tools from all databases
        all_tools = self._get_all_tools()

        # Create agent with all tools
        agent = create_sql_agent(
            llm=self.llm,
            db=primary_db,
            extra_tools=all_tools,
            agent_type="openai-tools",
            verbose=True,
            prefix=self._build_system_prompt(),
            max_iterations=10,
            handle_parsing_errors=True
        )

        logger.info("✅ Multi-database SQL agent created successfully")
        return agent

    async def query(
    self,
    user_message: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
        """Process a natural language query with retry logic"""

        if not self.agent:
            return {
                "content": "No databases are configured.",
                "sql_queries": [],
                "databases_used": []
            }

        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"Processing SQL query (attempt {attempt + 1}): {user_message[:100]}...")

                # Build input with chat history context
                input_text = user_message
                if chat_history:
                    history_text = "\n".join([
                        f"{msg['role']}: {msg['content']}"
                        for msg in chat_history[-4:]
                    ])
                    input_text = f"Previous conversation:\n{history_text}\n\nCurrent question: {user_message}"

                result = await self.agent.ainvoke({"input": input_text})
                output = result.get("output", "No results found.")

                logger.info("SQL agent query completed successfully")

                return {
                    "content": output,
                    "sql_queries": [],
                    "databases_used": list(self.databases.keys())
                }

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

                # Reconnect databases on connection error
                if "connection" in last_error.lower() or "ssl" in last_error.lower():
                    logger.info("Connection error detected - reconnecting databases...")
                    self._load_databases()
                    self.agent = self._create_multi_db_agent()
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Wait 1s before retry
                    continue

        # All retries failed
        logger.error(f"SQL agent failed after {max_retries} attempts: {last_error}")
        return {
            "content": f"I was unable to query the database after {max_retries} attempts. Please try again.",
            "sql_queries": [],
            "databases_used": []
        }

    def get_database_info(self) -> List[Dict[str, Any]]:
        """Get info about all connected databases"""
        info = []
        for db_name, db in self.databases.items():
            try:
                tables = db.get_usable_table_names()
                info.append({
                    "name": db_name,
                    "tables": tables,
                    "table_count": len(tables),
                    "status": "connected"
                })
            except Exception as e:
                info.append({
                    "name": db_name,
                    "tables": [],
                    "table_count": 0,
                    "status": f"error: {str(e)}"
                })
        return info


# Global instance
sql_agent_service = SQLAgentService()