import re
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage, AIMessage

class LASAnalyzerModel:
    def __init__(self, model, python_path):
        self.model = model
        self.python_path = python_path
        
    async def process_query(self, file_path, query):
        """Process a query about a LAS file using the LangChain agent and MCP tools."""
        async with MultiServerMCPClient() as client:
            await client.connect_to_server(
                "LAS File Analyzer",
                command=self.python_path,
                args=["file.py"],
                encoding_error_handler="ignore",
            )
            
            # Get all tools
            tools = client.get_tools()
            
            # Create the agent with debug enabled
            agent = create_react_agent(
                self.model, 
                tools, 
                debug=True
            )
            
            # Prepare the query
            full_query = f"Using the LAS file at {file_path}, {query}"
            
            # Process the query
            response = await agent.ainvoke(
                debug=True, 
                input={"messages": [HumanMessage(content=full_query)]}
            )
            
            # Extract thinking process
            thinking_process = self._extract_thinking_process(response)
            
            # Extract token usage
            token_usage = self._extract_token_usage(response)
            
            # Extract the response text
            response_text = self._extract_response_text(response)
            
            return {
                "response_text": response_text,
                "thinking_process": thinking_process,
                "token_usage": token_usage,
                "raw_response": response
            }
    
    def _extract_thinking_process(self, response):
        """Extract and format the thinking process from the response."""
        thinking_process = ""
        if isinstance(response, dict) and "messages" in response:
            for msg in response["messages"]:
                if hasattr(msg, "content") and isinstance(msg.content, list):
                    for content_item in msg.content:
                        if isinstance(content_item, dict) and content_item.get("type") == "thinking":
                            raw_thinking = content_item.get("thinking", "")
                            
                            # Split the raw thinking into paragraphs
                            paragraphs = [p for p in raw_thinking.split('\n\n') if p.strip()]
                            
                            # Format as numbered steps
                            numbered_steps = []
                            for i, paragraph in enumerate(paragraphs, 1):
                                # Clean up the paragraph - remove any existing numbering
                                clean_paragraph = re.sub(r'^\d+\.\s*', '', paragraph.strip())
                                numbered_steps.append(f"{i}. {clean_paragraph}")
                            
                            thinking_process = "\n\n".join(numbered_steps)
                            break  # Just use the first thinking block
        return thinking_process
    
    def _extract_token_usage(self, response):
        """Extract token usage information from the response."""
        token_usage = {
            "input": 0,
            "output": 0,
            "total": 0
        }
        
        if isinstance(response, dict) and "messages" in response:
            for msg in response["messages"]:
                if hasattr(msg, "usage_metadata"):
                    token_usage["input"] += msg.usage_metadata.get("input_tokens", 0)
                    token_usage["output"] += msg.usage_metadata.get("output_tokens", 0)
                    token_usage["total"] += msg.usage_metadata.get("total_tokens", 0)
        
        return token_usage
    
    def _extract_response_text(self, response):
        """Extract the response text from the response."""
        if isinstance(response, dict) and "messages" in response:
            ai_messages = [msg.content for msg in response["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                return ai_messages[-1]
        
        return "I couldn't process that request."
        
    def is_visualization_query(self, query):
        """Check if a query is related to visualization."""
        visualization_keywords = [
            "show", "plot", "display", "visualize", "visualization", "graph", 
            "chart", "track", "crossplot", "heatmap", "image", "picture", "draw"
        ]
        
        query_lower = query.lower()
        
        # Check for visualization keywords
        for keyword in visualization_keywords:
            if keyword in query_lower:
                return True
        
        return False
        
    def should_display_visualization(self, query, response):
        """Determine if a visualization should be displayed based on query and response."""
        # Check if query is visualization-related
        if self.is_visualization_query(query):
            return True
            
        # Check if response mentions visualization was created
        viz_created_phrases = [
            "created a visualization", "generated a plot", "created a plot",
            "visualization has been created", "plot has been generated",
            "created the following visualization", "generated the following plot",
            "here is the visualization", "here is the plot"
        ]
        
        for phrase in viz_created_phrases:
            if phrase.lower() in response.lower():
                return True
                
        return False
