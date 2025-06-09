import streamlit as st
import asyncio
import json
import logging
import os
import sqlite3
import pandas as pd
import time
from typing import Dict, List, Any, Optional
from fastmcp import Client
from datetime import datetime, timedelta
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Streamlit page config
st.set_page_config(
    page_title="MCP Client Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .tool-call {
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
    }
    .resource-read {
        background-color: #f0f8e7;
        border-left: 4px solid #28a745;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
    }
    .error-message {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
    }
</style>
""", unsafe_allow_html=True)

# LLM Models and Provider Selection
LLM_MODELS = ["openai", "anthropic", "ollama", "gemini", "bedrock"]
DATABASE_FILE = "mcp_chat_history.db"

# --- Database Schema and Operations ---
class ChatHistoryDB:
    def __init__(self, db_file: str = DATABASE_FILE):
        self.db_file = db_file
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with the chat history schema"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    llm_provider TEXT,
                    model_name TEXT,
                    parsing_mode TEXT,
                    user_query TEXT NOT NULL,
                    parsed_action TEXT,
                    tool_name TEXT,
                    resource_uri TEXT,
                    parameters TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    response_data TEXT,
                    formatted_response TEXT,
                    elapsed_time_ms INTEGER,
                    error_message TEXT,
                    success BOOLEAN NOT NULL DEFAULT 1
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON chat_history(session_id)")
            conn.commit()
    
    def insert_chat_entry(self, entry: Dict[str, Any]) -> int:
        """Insert a new chat entry and return the ID"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (
                    session_id, timestamp, llm_provider, model_name, parsing_mode,
                    user_query, parsed_action, tool_name, resource_uri, parameters,
                    confidence, reasoning, response_data, formatted_response,
                    elapsed_time_ms, error_message, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.get('session_id'),
                entry.get('timestamp'),
                entry.get('llm_provider'),
                entry.get('model_name'),
                entry.get('parsing_mode'),
                entry.get('user_query'),
                entry.get('parsed_action'),
                entry.get('tool_name'),
                entry.get('resource_uri'),
                entry.get('parameters'),
                entry.get('confidence'),
                entry.get('reasoning'),
                entry.get('response_data'),
                entry.get('formatted_response'),
                entry.get('elapsed_time_ms'),
                entry.get('error_message'),
                entry.get('success', True)
            ))
            
            entry_id = cursor.lastrowid
            conn.commit()
            return entry_id
    
    def get_chat_history(self, limit: int = 100, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve chat history with optional filters"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM chat_history"
            params = []
            
            if filters:
                conditions = []
                if filters.get('session_id'):
                    conditions.append("session_id = ?")
                    params.append(filters['session_id'])
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics from the database"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total queries
            cursor.execute("SELECT COUNT(*) FROM chat_history")
            stats['total_queries'] = cursor.fetchone()[0]
            
            if stats['total_queries'] == 0:
                return {
                    'total_queries': 0,
                    'success_rate': 0,
                    'avg_response_time_ms': 0,
                    'most_used_provider': 'None',
                    'most_used_tool': 'None',
                    'queries_last_24h': 0
                }
            
            # Success rate
            cursor.execute("SELECT COUNT(*) FROM chat_history WHERE success = 1")
            successful = cursor.fetchone()[0]
            stats['success_rate'] = (successful / stats['total_queries'] * 100)
            
            # Average response time
            cursor.execute("SELECT AVG(elapsed_time_ms) FROM chat_history WHERE elapsed_time_ms IS NOT NULL")
            avg_time = cursor.fetchone()[0]
            stats['avg_response_time_ms'] = avg_time if avg_time else 0
            
            # Most used LLM provider
            cursor.execute("""
                SELECT llm_provider, COUNT(*) as count 
                FROM chat_history 
                WHERE llm_provider IS NOT NULL 
                GROUP BY llm_provider 
                ORDER BY count DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            stats['most_used_provider'] = result[0] if result else "None"
            
            # Most used tool
            cursor.execute("""
                SELECT tool_name, COUNT(*) as count 
                FROM chat_history 
                WHERE tool_name IS NOT NULL 
                GROUP BY tool_name 
                ORDER BY count DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            stats['most_used_tool'] = result[0] if result else "None"
            
            # Recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM chat_history 
                WHERE timestamp >= datetime('now', '-1 day')
            """)
            stats['queries_last_24h'] = cursor.fetchone()[0]
            
            return stats

# Initialize session state
def init_session_state():
    if 'chat_history_db' not in st.session_state:
        st.session_state.chat_history_db = ChatHistoryDB()
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(f"{datetime.now()}{os.getpid()}".encode()).hexdigest()[:8]
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "anthropic"
    if 'use_llm' not in st.session_state:
        st.session_state.use_llm = True
    if 'server_connected' not in st.session_state:
        st.session_state.server_connected = False
    if 'available_tools' not in st.session_state:
        st.session_state.available_tools = []
    if 'available_resources' not in st.session_state:
        st.session_state.available_resources = []
    if 'mcp_client' not in st.session_state:
        st.session_state.mcp_client = None

# --- LLM Query Parser ---
class LLMQueryParser:
    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self.client = None
        self.model_name = None
        self.setup_llm_client()
    
    def setup_llm_client(self):
        """Setup the appropriate LLM client based on provider"""
        try:
            if self.provider == "openai":
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                    self.model_name = "gpt-4o-mini"
            
            elif self.provider == "anthropic":
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
                    self.model_name = "claude-3-5-sonnet-20241022"
            
            elif self.provider == "gemini":
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    self.client = genai.GenerativeModel("gemini-1.5-flash")
                    self.model_name = "gemini-1.5-flash"
                
        except Exception as e:
            logging.error(f"Failed to initialize {self.provider}: {e}")
            self.client = None
    
    def get_system_prompt(self, available_tools: List[Dict], available_resources: List[Dict] = None) -> str:
        """Create system prompt with available tools and resources"""
        tools_desc = "\n".join([
            f"- {tool['name']}: {tool.get('description', 'No description')}"
            for tool in available_tools
        ])
        
        resources_desc = ""
        if available_resources:
            resources_desc = "\n\nAvailable resources:\n" + "\n".join([
                f"- {resource['uri']}: {resource.get('description', 'No description')}"
                for resource in available_resources
            ])
        
        return f"""You are a tool and resource selection assistant. Given a user query, you must decide whether to use a tool, read a resource, or both.

Available tools:
{tools_desc}{resources_desc}

For each user query, respond with ONLY a JSON object in this exact format:
{{
    "action": "tool|resource|both",
    "tool": "tool_name_or_null",
    "resource_uri": "resource_uri_or_null",
    "params": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this action was chosen"
}}

Tool-specific parameter requirements:
- calculator: operation (add/subtract/multiply/divide/power), num1 (number), num2 (number)
- trig: operation (sine/cosine/tangent/arc sine/arc cosine/arc tangent), num1 (float), unit (degree/radian)
- stock_quote: ticker (stock symbol like AAPL, MSFT)
- health: no parameters needed
- echo: message (text to echo back)

Resource-specific patterns:
- info://server: Server information (no parameters)
- stock://{{ticker}}: Stock information for specific ticker

Remember: Respond with ONLY the JSON object, no additional text."""

    async def parse_query_with_llm(self, query: str, available_tools: List[Dict], available_resources: List[Dict] = None) -> Optional[Dict[str, Any]]:
        """Use LLM to parse the query"""
        if not self.client:
            return None
        
        system_prompt = self.get_system_prompt(available_tools, available_resources)
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.1,
                    max_tokens=300
                )
                llm_response = response.choices[0].message.content.strip()
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=300,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": query}]
                )
                llm_response = response.content[0].text.strip()
            
            elif self.provider == "gemini":
                response = self.client.generate_content(
                    f"{system_prompt}\n\nUser: {query}",
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 300
                    }
                )
                llm_response = response.text.strip()
            
            # Parse JSON response
            if llm_response.startswith("```json"):
                llm_response = llm_response.replace("```json", "").replace("```", "").strip()
            elif llm_response.startswith("```"):
                llm_response = llm_response.replace("```", "").strip()
            
            try:
                parsed_response = json.loads(llm_response)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    parsed_response = json.loads(json_match.group())
                else:
                    return None
            
            if parsed_response.get("action") and parsed_response.get("confidence", 0) >= 0.5:
                return parsed_response
            
        except Exception as e:
            logging.error(f"LLM parsing error: {e}")
        
        return None

# --- Rule-based Parser ---
class RuleBasedQueryParser:
    @staticmethod
    def parse_query(query: str) -> Optional[Dict[str, Any]]:
        import re
        query_lower = query.lower().strip()
        
        # Health check
        if any(word in query_lower for word in ["health", "status", "ping"]):
            return {"action": "tool", "tool": "health", "resource_uri": None, "params": {}}
        
        # Echo command
        if query_lower.startswith("echo "):
            return {"action": "tool", "tool": "echo", "resource_uri": None, "params": {"message": query[5:].strip()}}
        
        # Server info
        if any(word in query_lower for word in ["server info", "server information"]):
            return {"action": "resource", "tool": None, "resource_uri": "info://server", "params": {}}
        
        # Calculator
        calc_patterns = [
            ("add", ["plus", "add", "+", "sum"]),
            ("subtract", ["minus", "subtract", "-"]),
            ("multiply", ["times", "multiply", "*", "×"]),
            ("divide", ["divide", "divided by", "/"]),
            ("power", ["power", "to the power", "^"])
        ]
        
        for operation, keywords in calc_patterns:
            for keyword in keywords:
                if keyword in query_lower:
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
                    if len(numbers) >= 2:
                        return {
                            "action": "tool",
                            "tool": "calculator", 
                            "resource_uri": None,
                            "params": {"operation": operation, "num1": float(numbers[0]), "num2": float(numbers[1])}
                        }
        
        # Trig functions
        trig_patterns = [
            ("sine", ["sine", "sin"]),
            ("cosine", ["cosine", "cos"]),
            ("tangent", ["tangent", "tan"])
        ]
        
        for operation, keywords in trig_patterns:
            for keyword in keywords:
                if keyword in query_lower:
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
                    if numbers:
                        unit = "radian" if any(word in query_lower for word in ["radian", "rad"]) else "degree"
                        return {
                            "action": "tool",
                            "tool": "trig",
                            "resource_uri": None,
                            "params": {"operation": operation, "num1": float(numbers[0]), "unit": unit}
                        }
        
        # Stock queries
        company_mapping = {
            "apple": "AAPL", "google": "GOOGL", "microsoft": "MSFT", "tesla": "TSLA",
            "amazon": "AMZN", "meta": "META", "nvidia": "NVDA"
        }
        
        is_info_request = any(keyword in query_lower for keyword in ["about", "company", "info", "what is"])
        is_price_request = any(keyword in query_lower for keyword in ["stock", "price", "quote", "trading"])
        
        # Check for ticker or company name
        tickers = re.findall(r'\b[A-Z]{2,5}\b', query.upper())
        excluded = {"GET", "THE", "FOR", "AND", "BUT", "NOT", "YOU", "ALL", "CAN", "STOCK", "PRICE"}
        valid_tickers = [t for t in tickers if t not in excluded]
        
        for company, ticker in company_mapping.items():
            if company in query_lower:
                valid_tickers.append(ticker)
        
        if valid_tickers:
            ticker = valid_tickers[0]
            if is_info_request and is_price_request:
                return {"action": "both", "tool": "stock_quote", "resource_uri": f"stock://{ticker}", "params": {"ticker": ticker}}
            elif is_info_request:
                return {"action": "resource", "tool": None, "resource_uri": f"stock://{ticker}", "params": {}}
            elif is_price_request:
                return {"action": "tool", "tool": "stock_quote", "resource_uri": None, "params": {"ticker": ticker}}
        
        return None

# --- Utility Functions ---
def extract_result_data(result):
    """Extract actual data from FastMCP result object"""
    try:
        if isinstance(result, list) and len(result) > 0:
            content_item = result[0]
            if hasattr(content_item, 'text'):
                try:
                    return json.loads(content_item.text)
                except json.JSONDecodeError:
                    return {"text": content_item.text}
            else:
                return {"content": str(content_item)}
        elif hasattr(result, 'content') and result.content:
            content_item = result.content[0]
            if hasattr(content_item, 'text'):
                try:
                    return json.loads(content_item.text)
                except json.JSONDecodeError:
                    return {"text": content_item.text}
            else:
                return {"content": str(content_item)}
        else:
            return result if isinstance(result, dict) else {"result": str(result)}
    except Exception as e:
        return {"error": f"Could not parse result: {e}"}

def extract_resource_data(result):
    """Extract data from resource result"""
    try:
        if isinstance(result, list) and len(result) > 0:
            content_item = result[0]
            return content_item.text if hasattr(content_item, 'text') else str(content_item)
        elif hasattr(result, 'content') and result.content:
            content_item = result.content[0]
            return content_item.text if hasattr(content_item, 'text') else str(content_item)
        else:
            return str(result)
    except Exception as e:
        return f"Could not parse resource: {e}"

def format_result_for_display(tool_name: str, result: Dict) -> str:
    """Format tool results for Streamlit display"""
    if isinstance(result, dict) and "error" in result:
        return f"❌ **Error:** {result['error']}"
    
    if tool_name == "calculator":
        expression = result.get('expression', f"{result.get('num1', '?')} {result.get('operation', '?')} {result.get('num2', '?')} = {result.get('result', '?')}")
        return f"🧮 **Calculator:** {expression}"
    
    elif tool_name == "trig":
        expression = result.get('expression', f"{result.get('operation', '?')}({result.get('num1', '?')}) = {result.get('result', '?')}")
        return f"📐 **Trigonometry:** {expression}"
    
    elif tool_name == "stock_quote":
        if "current_price" in result:
            ticker = result.get('ticker', 'Unknown')
            name = result.get('company_name', ticker)
            price = result.get('current_price', 0)
            currency = result.get('currency', 'USD')
            return f"📈 **{name} ({ticker}):** {currency} {price}"
        else:
            return f"❌ **Stock Error:** {result.get('error', 'Unknown error')}"
    
    elif tool_name == "health":
        return f"✅ **Health:** {result.get('message', 'Server is healthy')}"
    
    elif tool_name == "echo":
        return f"🔊 **Echo:** {result.get('echo', result.get('message', str(result)))}"
    
    return f"✅ **Result:** {json.dumps(result, indent=2)}"

def format_resource_for_display(resource_uri: str, content: str) -> str:
    """Format resource results for Streamlit display"""
    if resource_uri.startswith("info://server"):
        return f"🖥️ **Server Info:** {content}"
    elif resource_uri.startswith("stock://"):
        ticker = resource_uri.split("://")[1]
        return f"🏢 **Company Info ({ticker}):** {content}"
    else:
        return f"📄 **Resource ({resource_uri}):** {content}"

# --- Async MCP Operations ---
async def connect_to_mcp_server():
    """Connect to MCP server and discover tools/resources"""
    try:
        client = Client("mcp_server.py")
        await client.__aenter__()
        
        # Discover tools
        tools = await client.list_tools()
        available_tools = [{"name": tool.name, "description": tool.description} for tool in tools] if tools else []
        
        # Discover resources
        try:
            resources = await client.list_resources()
            available_resources = [{"uri": resource.uri, "description": resource.description} for resource in resources] if resources else []
        except:
            available_resources = []
        
        # Add dynamic resources
        available_resources.append({"uri": "stock://{ticker}", "description": "Stock company information for any ticker"})
        
        return client, available_tools, available_resources
    except Exception as e:
        st.error(f"Failed to connect to MCP server: {e}")
        return None, [], []

async def execute_mcp_query(client, parsed_query):
    """Execute MCP tool calls and resource reads"""
    results = []
    start_time = time.time()
    
    action = parsed_query.get("action")
    tool_name = parsed_query.get("tool")
    resource_uri = parsed_query.get("resource_uri")
    parameters = parsed_query.get("params", {})
    
    # Execute tool call if needed
    if action in ["tool", "both"] and tool_name:
        try:
            tool_result = await client.call_tool(tool_name, parameters)
            tool_data = extract_result_data(tool_result)
            results.append({
                "type": "tool",
                "name": tool_name,
                "data": tool_data,
                "formatted": format_result_for_display(tool_name, tool_data),
                "success": "error" not in tool_data
            })
        except Exception as e:
            results.append({
                "type": "error",
                "message": f"Tool call error: {e}",
                "success": False
            })
    
    # Execute resource read if needed
    if action in ["resource", "both"] and resource_uri:
        try:
            resource_result = await client.read_resource(resource_uri)
            resource_content = extract_resource_data(resource_result)
            results.append({
                "type": "resource",
                "uri": resource_uri,
                "content": resource_content,
                "formatted": format_resource_for_display(resource_uri, resource_content),
                "success": True
            })
        except Exception as e:
            results.append({
                "type": "error",
                "message": f"Resource read error: {e}",
                "success": False
            })
    
    elapsed_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
    return results, elapsed_time

# --- Main App ---
def main():
    init_session_state()
    
    # Create tabs for different pages
    tab1, tab2, tab3 = st.tabs(["🚀 Chat Interface", "📊 History & Analytics", "⚙️ Database Management"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        history_analytics()
    
    with tab3:
        database_management()

def chat_interface():
    """Main chat interface page"""
    # Header
    st.markdown('<h1 class="main-header">🚀 MCP Client Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Session info
        st.info(f"📍 **Session ID:** `{st.session_state.session_id}`")
        
        # LLM Provider Selection
        st.session_state.llm_provider = st.selectbox(
            "🤖 LLM Provider",
            LLM_MODELS,
            index=LLM_MODELS.index(st.session_state.llm_provider),
            help="Choose your preferred LLM provider"
        )
        
        # Parsing Mode
        st.session_state.use_llm = st.checkbox(
            "🧠 Use LLM Parsing",
            value=st.session_state.use_llm,
            help="Use LLM for intelligent query parsing vs rule-based parsing"
        )
        
        # API Keys Status
        st.subheader("🔑 API Keys Status")
        api_keys_status = {
            "OpenAI": "✅" if os.getenv("OPENAI_API_KEY") else "❌",
            "Anthropic": "✅" if os.getenv("ANTHROPIC_API_KEY") else "❌",
            "Gemini": "✅" if os.getenv("GEMINI_API_KEY") else "❌",
            "AWS": "✅" if os.getenv("AWS_ACCESS_KEY_ID") else "❌"
        }
        
        for provider, status in api_keys_status.items():
            st.write(f"{status} {provider}")
        
        # Server Connection
        st.subheader("🔌 Server Status")
        if st.button("🔄 Connect to MCP Server"):
            with st.spinner("Connecting to MCP server..."):
                try:
                    # Create new event loop for async operations
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Run the connection in the new loop
                    client, tools, resources = loop.run_until_complete(connect_to_mcp_server())
                    
                    if client:
                        st.session_state.server_connected = True
                        st.session_state.available_tools = tools
                        st.session_state.available_resources = resources
                        st.session_state.mcp_client = client
                        st.success("✅ Connected to MCP server!")
                    else:
                        st.session_state.server_connected = False
                        st.error("❌ Failed to connect to MCP server")
                except Exception as e:
                    st.session_state.server_connected = False
                    st.error(f"❌ Connection error: {e}")
        
        # Connection Status Display
        if st.session_state.server_connected:
            st.success("🟢 Server Connected")
        else:
            st.error("🔴 Server Disconnected")
        
        # Show tools and resources only if connected (FIXED: Remove duplicate display)
        if st.session_state.server_connected and st.session_state.available_tools:
            with st.expander("🔧 Available Tools"):
                for tool in st.session_state.available_tools:
                    st.write(f"• **{tool['name']}**: {tool['description']}")
        
        if st.session_state.server_connected and st.session_state.available_resources:
            with st.expander("📚 Available Resources"):
                for resource in st.session_state.available_resources:
                    st.write(f"• **{resource['uri']}**: {resource['description']}")
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Query Interface")
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            **🧮 Calculator:**
            - "What's 15 plus 27?"
            - "Multiply 12 by 8"
            - "Divide 100 by 4"
            
            **📐 Trigonometry:**
            - "Find sine of 30 degrees"
            - "Calculate cosine of pi/4 radians"
            
            **📈 Stock Data:**
            - "Get Apple stock price"
            - "Tell me about Tesla as a company"
            - "Apple stock price and company info"
            
            **🔧 System:**
            - "Server health check"
            - "Echo hello world"
            - "Server information"
            """)
        
        # Query Input
        user_query = st.text_input(
            "🎯 Enter your query:",
            placeholder="What's 15 plus 27?",
            key="user_query_input"
        )
        
        col_submit, col_clear = st.columns([1, 1])
        with col_submit:
            submit_button = st.button("🚀 Submit Query", type="primary")
        with col_clear:
            clear_button = st.button("🗑️ Clear Session")
        
        if clear_button:
            st.session_state.session_id = hashlib.md5(f"{datetime.now()}{os.getpid()}".encode()).hexdigest()[:8]
            st.success("✅ New session started!")
            time.sleep(1)  # Brief pause before rerun
            st.rerun()
        
        # Process Query
        if submit_button and user_query:
            if not st.session_state.server_connected:
                st.error("❌ Please connect to the MCP server first!")
            else:
                # Create a placeholder for the processing status
                status_placeholder = st.empty()
                results_placeholder = st.empty()
                
                try:
                    status_placeholder.info("🤖 Processing query...")
                    
                    # Initialize parser
                    parsed_query = None
                    model_name = None
                    
                    if st.session_state.use_llm:
                        parser = LLMQueryParser(st.session_state.llm_provider)
                        if parser.client:
                            # Create new event loop for LLM parsing
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                parsed_query = loop.run_until_complete(
                                    parser.parse_query_with_llm(
                                        user_query,
                                        st.session_state.available_tools,
                                        st.session_state.available_resources
                                    )
                                )
                                model_name = parser.model_name
                            finally:
                                loop.close()  # Important: Close the loop
                        else:
                            status_placeholder.warning("🔄 LLM not available, using rule-based parsing")
                            parsed_query = RuleBasedQueryParser.parse_query(user_query)
                    else:
                        parsed_query = RuleBasedQueryParser.parse_query(user_query)
                    
                    if parsed_query:
                        status_placeholder.info("⚡ Executing query...")
                        
                        # Execute query
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            results, elapsed_time = loop.run_until_complete(
                                execute_mcp_query(st.session_state.mcp_client, parsed_query)
                            )
                        finally:
                            loop.close()  # Important: Close the loop
                        
                        # Determine overall success
                        overall_success = all(result.get('success', True) for result in results)
                        error_messages = [result.get('message', '') for result in results if result.get('type') == 'error']
                        
                        # Prepare data for database
                        db_entry = {
                            'session_id': st.session_state.session_id,
                            'timestamp': datetime.now(),
                            'llm_provider': st.session_state.llm_provider if st.session_state.use_llm else None,
                            'model_name': model_name,
                            'parsing_mode': 'LLM' if st.session_state.use_llm else 'Rule-based',
                            'user_query': user_query,
                            'parsed_action': parsed_query.get('action'),
                            'tool_name': parsed_query.get('tool'),
                            'resource_uri': parsed_query.get('resource_uri'),
                            'parameters': json.dumps(parsed_query.get('params', {})),
                            'confidence': parsed_query.get('confidence'),
                            'reasoning': parsed_query.get('reasoning'),
                            'response_data': json.dumps([r for r in results if r.get('type') != 'error']),
                            'formatted_response': '\n'.join([r.get('formatted', '') for r in results if r.get('formatted')]),
                            'elapsed_time_ms': elapsed_time,
                            'error_message': '; '.join(error_messages) if error_messages else None,
                            'success': overall_success
                        }
                        
                        # Save to database
                        entry_id = st.session_state.chat_history_db.insert_chat_entry(db_entry)
                        
                        # Clear status and show results
                        status_placeholder.empty()
                        
                        with results_placeholder.container():
                            st.success(f"✅ Query processed in {elapsed_time}ms (Entry ID: {entry_id})")
                            
                            # Display results
                            for result in results:
                                if result['type'] == 'tool':
                                    st.markdown(f'<div class="tool-call">{result["formatted"]}</div>', unsafe_allow_html=True)
                                elif result['type'] == 'resource':
                                    st.markdown(f'<div class="resource-read">{result["formatted"]}</div>', unsafe_allow_html=True)
                                elif result['type'] == 'error':
                                    st.markdown(f'<div class="error-message">❌ {result["message"]}</div>', unsafe_allow_html=True)
                        
                        # Force a small delay to ensure UI updates
                        time.sleep(0.1)
                        
                    else:
                        status_placeholder.error("❓ I couldn't understand your query. Please try rephrasing.")
                        
                except Exception as e:
                    status_placeholder.error(f"❌ Error processing query: {e}")
                    logging.error(f"Query processing error: {e}", exc_info=True)
    
    with col2:
        st.subheader("📊 Query Analysis")
        
        # Get recent entries for this session
        try:
            recent_entries = st.session_state.chat_history_db.get_chat_history(
                limit=5, 
                filters={'session_id': st.session_state.session_id}
            )
            
            if recent_entries:
                latest_entry = recent_entries[0]
                
                # Parser info
                st.info(f"🔍 **Parser:** {latest_entry['parsing_mode']}")
                if latest_entry['model_name']:
                    st.info(f"🤖 **Model:** {latest_entry['model_name']}")
                
                # Parsed query details
                try:
                    analysis_data = {
                        "action": latest_entry['parsed_action'],
                        "tool": latest_entry['tool_name'],
                        "resource_uri": latest_entry['resource_uri'],
                        "confidence": latest_entry['confidence'],
                        "elapsed_time_ms": latest_entry['elapsed_time_ms']
                    }
                    st.json(analysis_data)
                except Exception as e:
                    st.error(f"Error displaying analysis: {e}")
                
                # Session stats
                try:
                    session_stats = st.session_state.chat_history_db.get_chat_history(
                        limit=1000, 
                        filters={'session_id': st.session_state.session_id}
                    )
                    
                    if len(session_stats) > 1:
                        successful = sum(1 for entry in session_stats if entry['success'])
                        avg_time = sum(entry['elapsed_time_ms'] or 0 for entry in session_stats) / len(session_stats)
                        
                        st.markdown("**Session Statistics:**")
                        st.metric("Queries", len(session_stats))
                        st.metric("Success Rate", f"{(successful/len(session_stats)*100):.1f}%")
                        st.metric("Avg Response Time", f"{avg_time:.0f}ms")
                except Exception as e:
                    st.error(f"Error calculating session stats: {e}")
            else:
                st.info("💡 No queries in this session yet. Try asking something!")
                
        except Exception as e:
            st.error(f"Error loading query analysis: {e}")

def history_analytics():
    """History and analytics page"""
    st.header("📊 Chat History & Analytics")
    
    try:
        # Quick stats
        stats = st.session_state.chat_history_db.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", stats['total_queries'])
        with col2:
            st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
        with col3:
            st.metric("Avg Response Time", f"{stats['avg_response_time_ms']:.0f}ms")
        with col4:
            st.metric("Queries (24h)", stats['queries_last_24h'])
        
        # Simple history display
        st.subheader("📋 Recent Query History")
        
        # Get recent history
        history = st.session_state.chat_history_db.get_chat_history(limit=50)
        
        if history:
            # Convert to DataFrame for better display
            df = pd.DataFrame(history)
            
            # Format timestamp for display
            df['formatted_timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Select important columns to display
            display_columns = [
                'id', 'formatted_timestamp', 'session_id', 'parsing_mode', 
                'user_query', 'tool_name', 'success', 'elapsed_time_ms'
            ]
            
            # Filter columns that exist
            available_columns = [col for col in display_columns if col in df.columns]
            display_df = df[available_columns].copy()
            
            # Format success column
            if 'success' in display_df.columns:
                display_df['success'] = display_df['success'].map({True: '✅', False: '❌'})
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
        else:
            st.info("No query history available yet.")
            
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

def database_management():
    """Database management page"""
    st.header("⚙️ Database Management")
    
    try:
        # Database info
        st.subheader("📊 Database Information")
        
        stats = st.session_state.chat_history_db.get_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Database File:** `{DATABASE_FILE}`
            **Total Records:** {stats['total_queries']}
            **Most Used Provider:** {stats['most_used_provider']}
            **Most Used Tool:** {stats['most_used_tool']}
            """)
        
        with col2:
            # Database file size
            try:
                if os.path.exists(DATABASE_FILE):
                    file_size = os.path.getsize(DATABASE_FILE) / 1024  # KB
                    st.markdown(f"""
                    **File Size:** {file_size:.2f} KB
                    **Success Rate:** {stats['success_rate']:.1f}%
                    **Avg Response Time:** {stats['avg_response_time_ms']:.0f}ms
                    **Recent Activity:** {stats['queries_last_24h']} queries (24h)
                    """)
                else:
                    st.info("Database file will be created on first query.")
            except Exception as e:
                st.warning(f"Could not read database file info: {e}")
        
        # Export functionality
        st.subheader("📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to CSV"):
                try:
                    history = st.session_state.chat_history_db.get_chat_history(limit=10000)
                    if history:
                        df = pd.DataFrame(history)
                        csv = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"mcp_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No data to export.")
                except Exception as e:
                    st.error(f"Export error: {e}")
        
        with col2:
            if st.button("📋 Export to JSON"):
                try:
                    history = st.session_state.chat_history_db.get_chat_history(limit=10000)
                    if history:
                        json_data = json.dumps(history, indent=2, default=str)
                        
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=f"mcp_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    else:
                        st.warning("No data to export.")
                except Exception as e:
                    st.error(f"Export error: {e}")
        
        # Cleanup functionality
        st.subheader("🧹 Database Cleanup")
        
        if st.button("🔥 Clear All Data", type="secondary"):
            try:
                with sqlite3.connect(DATABASE_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM chat_history")
                    conn.commit()
                
                st.success("✅ All data cleared.")
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"Clear error: {e}")
                
    except Exception as e:
        st.error(f"Database management error: {e}")

if __name__ == "__main__":
    main()