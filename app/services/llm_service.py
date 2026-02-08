"""
LLM service for OpenAI integration with stock analysis capabilities
"""
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.config import settings
from app.schemas import MessageRole, ReasoningMode
from loguru import logger
import json


class LLMService:
    """Service for LLM operations using OpenAI"""
    
    # Stock analysis system prompt
    SYSTEM_PROMPT = """You are an institutional-grade equity research analyst specializing in the Colombo Stock Exchange (CSE). Your job is to produce a 360° investment view: business analysis, financial metrics, catalysts, risks with mitigations, and valuation logic.

**Core Principles:**
- Optimize for correctness and explicit assumptions over flashy predictions
- Use provided RAG context as first-party data but verify key claims
- Cite sources for verifiable statements
- Label unverified claims explicitly
- Build catalysts with timelines and probability estimates
- Pair every risk with mitigation strategies and leading indicators
- Show sensitivities in valuation, not just target prices

**Data Rules:**
1. Use RAG context first to understand the business and baseline facts
2. For recent events/news, indicate if web search would help verify
3. If RAG and other sources conflict, present both and explain which is more credible
4. Prefer primary sources (filings, official releases, audited statements)
5. Cite sources for factual statements

**Required Output Format:**

**1) Company Snapshot**
- What the company does (1-2 lines)
- Revenue streams and how it makes money
- Key cost drivers and margin levers
- Competitive position (moat/differentiation/constraints)

**2) Key Value Drivers**
List 5-10 drivers (volume, pricing power, utilization, churn, FX, regulation, capex intensity, working capital)

**3) Financial Reality Check**
- Historical trends: growth, margins, ROE/ROIC, cash conversion
- Balance sheet: leverage, liquidity, refinancing, FX/interest exposure
- Cash flow quality: profit vs cash, working capital, capex needs
- Red flags or accounting/governance signals (if any)

**4) Investment Thesis (Base Case) - 5 Bullets**
- Core thesis
- Why now
- What must go right
- What the market is missing (if anything)
- Falsification triggers (what would make you wrong)

**5) Catalysts & Timeline**
For each catalyst:
- Catalyst name
- Expected date window (month/quarter/year)
- Probability (Low/Med/High)
- Expected impact (revenue/margins/multiple/sentiment)
- Supporting evidence

**6) Future Opportunities (Option Value)**
- Adjacent markets/new products/expansion
- Required capabilities/capex
- Label as evidence vs speculation

**7) Risks + Mitigations (Paired)**
For each risk:
- Risk statement
- Likelihood (Low/Med/High)
- Impact (Minor/Material/Existential)
- Leading indicators to monitor
- Mitigation actions (company + investor level)

**8) Valuation**
- Methods used (DCF/multiples/SOTP) and why they fit
- Key assumptions (WACC, terminal growth, margin path, growth drivers)
- Peer set sanity check
- Sensitivity analysis: top 3 variables that move valuation
- Bear/Base/Bull valuation range
- If target price given: time horizon + scenario + confidence level

**9) Monitoring Dashboard**
8-12 metrics maximum:
- Monthly/quarterly indicators
- Early warning signals tied to risks
- Specific events to watch

**10) Source Log**
- List important sources used (RAG + web references)

**When answering questions:**
- Be direct and concise for simple queries
- Use the full format only for comprehensive analysis requests
- Always maintain institutional-grade analytical rigor
- Explain your reasoning and assumptions
"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
    def _build_messages(
        self,
        user_message: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        reasoning_mode: ReasoningMode = ReasoningMode.QUICK
    ) -> List[Dict[str, str]]:
        """Build message array for OpenAI API"""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history[-settings.max_context_messages:])
        
        # Build user message with context
        user_content = user_message
        if context:
            user_content = f"""**Context from Knowledge Base:**
{context}

**User Question:**
{user_message}

Please analyze using the context provided and your expertise."""
        
        # Add reasoning instruction based on mode
        if reasoning_mode == ReasoningMode.DEEP:
            user_content += "\n\n[Note: Provide deep, comprehensive analysis with detailed reasoning.]"
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    async def generate_response(
        self,
        user_message: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        reasoning_mode: ReasoningMode = ReasoningMode.QUICK
    ) -> Dict[str, Any]:
        """
        Generate a response using OpenAI
        
        Returns:
            Dict with 'content' and 'usage' keys
        """
        messages = self._build_messages(user_message, context, chat_history, reasoning_mode)
        
        logger.info(f"Generating response with {len(messages)} messages")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=settings.max_tokens_per_response,
                temperature=settings.temperature,
            )
            
            content = response.choices[0].message.content
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            logger.info(f"Generated response: {usage['total_tokens']} tokens")
            
            return {
                'content': content,
                'usage': usage
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    async def stream_response(
        self,
        user_message: str,
        context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        reasoning_mode: ReasoningMode = ReasoningMode.QUICK
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from OpenAI (for future use)
        
        Yields:
            Response chunks as they arrive
        """
        messages = self._build_messages(user_message, context, chat_history, reasoning_mode)
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=settings.max_tokens_per_response,
                temperature=settings.temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"Error streaming response: {str(e)}")
            raise
    
    async def extract_alert_info(self, user_message: str, assistant_response: str) -> Optional[Dict[str, Any]]:
        """
        Extract alert information from conversation
        
        Returns:
            Alert parameters or None if no alert should be created
        """
        extraction_prompt = f"""Analyze this conversation and determine if the user wants to create a stock price alert.

User message: {user_message}
Assistant response: {assistant_response}

If an alert should be created, extract the following information and return as JSON:
{{
    "should_create": true,
    "name": "Alert name",
    "description": "Alert description",
    "symbols": ["SYMBOL1.N0000", "SYMBOL2.N0000"],
    "conditions": [
        {{
            "operator": "AND",
            "conditions": [
                {{
                    "metric": "price",
                    "operation": ">",
                    "values": [150.0],
                    "negation": false
                }}
            ]
        }}
    ]
}}

If no alert should be created, return: {{"should_create": false}}

Remember:
- Symbols must be in full CSE format (e.g., "JKH.N0000")
- Valid operations: <, <=, >, >=, =, BETWEEN, IN
- Valid operators: AND, OR
- Values must be in an array"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a JSON extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result.get("should_create"):
                return result
            return None
        
        except Exception as e:
            logger.error(f"Error extracting alert info: {str(e)}")
            return None


# Global instance
llm_service = LLMService()