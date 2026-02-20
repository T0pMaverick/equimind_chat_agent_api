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

    def _load_databases(self):
        """Load all configured PostgreSQL databases"""
        db_configs = [
            (settings.db_1_url, settings.db_1_name),
            (settings.db_2_url, settings.db_2_name),
            (settings.db_3_url, settings.db_3_name),
        ]

        for db_url, db_name in db_configs:
            if db_url:
                try:
                    db = SQLDatabase.from_uri(
                        db_url,
                        sample_rows_in_table_info=0,  # ✅ Disable sample rows - prevents SSL timeout
                        engine_args={
                            "pool_pre_ping": True,        # ✅ Test connection before use
                            "pool_recycle": 300,           # ✅ Recycle connections every 5 mins
                            "pool_timeout": 30,            # ✅ Wait max 30s for connection
                            "connect_args": {
                                "connect_timeout": 10,     # ✅ Connection timeout
                                "keepalives": 1,           # ✅ Enable TCP keepalives
                                "keepalives_idle": 30,     # ✅ Start keepalive after 30s idle
                                "keepalives_interval": 10, # ✅ Send keepalive every 10s
                                "keepalives_count": 5      # ✅ Retry 5 times before dropping
                            }
                        }
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
You have access to {len(self.databases)} PostgreSQL databases:

{db_info}

**Your Responsibilities:**
1. Understand the user's question in natural language
2. Identify which database(s) contain the relevant data
3. Generate accurate PostgreSQL SQL queries
4. Execute the queries and return clear, formatted results
5. If data spans multiple databases, query each one and combine the results

**Rules:**
- Always use SELECT queries only (never INSERT, UPDATE, DELETE, DROP)
- Limit results to 50 rows unless user asks for more
- Format numbers with proper units (millions, billions)
- If a query fails, try an alternative approach
- Always explain what data you found
- For stock symbols, try both short (SAMP) and full format (SAMP.N0000)

**Response Format:**
- Answer the question directly
- Show the key data points
- Add brief interpretation where helpful
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