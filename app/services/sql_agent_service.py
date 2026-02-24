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
            "trades": """
                Individual trade transactions.
                Use for: trade-level analysis, execution prices
            """,
            
            "orderbook_snapshots": """
                Order book state at different times.
                Use for: market depth, bid/ask analysis
            """,
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

    When user asks for:
    - "September 2025" → WHERE reporting_period = '2025_September'
    - "Q3 2025" → WHERE reporting_period IN ('2025_July', '2025_August', '2025_September')
    - "2025 data" → WHERE reporting_period LIKE '2025_%'

    **CRITICAL COLUMN NAMES:**
    - Use `company_code` (NOT company_name) for company identifier
    - Column names: dividend_yield, earnings_per_share, trailing_eps, return_on_equity, etc.

    **TABLE SELECTION RULES:**

    1. For DIVIDEND YIELD queries:
    → Use `companies_financial_data.dividend_yield`

    2. For FINANCIAL RATIOS:
    → Use `companies_financial_data` table

    3. For PRICE DATA:
    → Use `historical_stock_data` table

    **Rules:**
    - SELECT only (never INSERT, UPDATE, DELETE)
    - Limit to 50 rows unless specified
    - If query fails, try alternative column names
    - Always check reporting_period format: 'YYYY_MonthName'
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