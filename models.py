import os
import re

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

MODEL_COST_PER_1K_INPUT_TOKENS = {
    "claude-3-sonnet-20240229": 0.003,
    "claude-3-5-sonnet-20240620": 0.003,
    "claude-3-5-sonnet-20241022": 0.003,
    "claude-3-7-sonnet-20250219": 0.003,
    "claude-3-haiku-20240307": 0.00025,
    "claude-3-opus-20240229": 0.015,
    "claude-3-5-haiku-20241022": 0.0008,
}

MODEL_COST_PER_1K_OUTPUT_TOKENS = {
    "claude-3-sonnet-20240229": 0.015,
    "claude-3-5-sonnet-20240620": 0.015,
    "claude-3-5-sonnet-20241022": 0.015,
    "claude-3-7-sonnet-20250219": 0.015,
    "claude-3-haiku-20240307": 0.00125,
    "claude-3-opus-20240229": 0.075,
    "claude-3-5-haiku-20241022": 0.004,
}


class LASAnalyzerModel:
    def __init__(self, model, python_path):
        # Currently self.model expects a ChatAnthropic
        # TODO: Make more generic so that ChatOpenAI works too in the future
        self.model: ChatAnthropic = model
        self.python_path = python_path
        self.client = None
        self.agent = None
        self._initialized = False
        self._file_paths = set()  # Track valid file paths
        
        # System message that doesn't change between queries
        self.system_message = """
        When responding to queries about LAS files, if you create a visualization,
        explicitly indicate this in your response with a special tag: [VISUALIZATION:filename].
        Only include this tag if you've actually created a visualization.
        """
    
    async def initialize(self):
        """Initialize the MCP client and React agent."""
        if not self._initialized:
            # Create the MCP client
            self.client = MultiServerMCPClient()
            await self.client.connect_to_server(
                "LAS File Analyzer",
                command=self.python_path,
                args=["file.py"],
                encoding_error_handler="ignore",
            )
            
            # Create the agent with debug enabled
            self.agent = create_react_agent(self.model, self.client.get_tools(), debug=True)
            
            self._initialized = True
            return True
        return False
    
    async def cleanup(self):
        """Clean up resources when the model is no longer needed."""
        if self.client and self._initialized:
            await self.client.close()
            self.client = None
            self.agent = None
            self._initialized = False
            return True
        return False
    
    def register_file_path(self, file_path):
        """Register a file path as valid for processing."""
        if os.path.exists(file_path):
            self._file_paths.add(file_path)
            return True
        return False
    
    def validate_file_path(self, file_path):
        """Check if a file path is valid and registered."""
        return file_path in self._file_paths or os.path.exists(file_path)

    async def process_query(self, file_path, query):
        """Process a query about a LAS file using the LangChain agent and MCP tools."""
        # Ensure the client and agent are initialized
        if not self._initialized:
            await self.initialize()
        
        # Validate file path
        if not self.validate_file_path(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}",
                "thinking_process": "",
                "tool_messages": [],
                "token_usage": {"input": 0, "output": 0, "total": 0},
                "token_cost": {"input": 0, "output": 0, "total": 0},
                "should_display_viz": False,
                "viz_info": None,
            }
        
        # Register the file path if it exists but wasn't registered
        if file_path not in self._file_paths and os.path.exists(file_path):
            self.register_file_path(file_path)

        # Prepare the query
        full_query = f"Using the LAS file at {file_path}, {query}"

        # Process the query with the system message
        response = await self.agent.ainvoke(
            debug=True,
            input={
                "messages": [
                    SystemMessage(content=self.system_message),
                    HumanMessage(content=full_query),
                ]
            },
        )

        # Extract message components (thinking, text, and tools) in a single pass
        components = self._extract_message_components(response)
        thinking_process = components["thinking_process"]
        tool_messages = components["tool_messages"]
        response_text = components["response_text"]
        all_text_contents = components["all_text_contents"]

        # Extract token usage and cost
        token_usage, token_cost = self._extract_token_usage_and_cost(response)

        # Extract visualization information
        viz_info = self._extract_visualization_info(response_text)

        return {
            "status": "success",
            "response_text": viz_info["clean_response"],
            "thinking_process": thinking_process,
            "tool_messages": tool_messages,
            "token_usage": token_usage,
            "token_cost": token_cost,
            "raw_response": response,
            "should_display_viz": viz_info["should_display"],
            "viz_info": viz_info["viz_info"],
            "all_text_contents": all_text_contents,
        }
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    def _extract_message_components(self, response):
        result = {
            "thinking_process": "",
            "response_text": "",
            "tool_messages": [],
            "all_text_contents": []
        }
        
        if not isinstance(response, dict) or "messages" not in response:
            return result
            
        for msg in response["messages"]:
            # Handle AIMessage
            if isinstance(msg, AIMessage):
                # Handle structured content (list of dictionaries)
                if isinstance(msg.content, list):
                    for item in msg.content:
                        if not isinstance(item, dict):
                            continue
                            
                        item_type = item.get("type")
                        
                        # Extract thinking
                        if item_type == "thinking":
                            result["thinking_process"] += "\n\n" + item.get("thinking", "")
                        
                        # Extract text
                        elif item_type == "text":
                            text_content = item.get("text", "")
                            result["all_text_contents"].append(text_content)
                        
                        # Extract tool_use
                        elif item_type in ["tool", "tool_use"]:
                            result["tool_messages"].append({
                                "name": item.get("name", "unknown_tool"),
                                "input": item.get("input", {}),
                                "output": item.get("output", "No output"),
                                "id": item.get("id", "")
                            })
                
                # Handle simple string content
                elif isinstance(msg.content, str):
                    result["all_text_contents"].append(msg.content)
                
                # Check for tool calls in additional_kwargs
                if hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs:
                    for tool_call in msg.additional_kwargs["tool_calls"]:
                        result["tool_messages"].append({
                            "name": tool_call.get("name", "unknown_tool"),
                            "input": tool_call.get("args", {}),
                            "id": tool_call.get("id", ""),
                            "output": "No output"
                        })
            
            # Handle ToolMessage
            elif hasattr(msg, "type") and msg.type == "tool":
                # Find the corresponding tool in our list by id
                tool_id = getattr(msg, "tool_call_id", None)
                if tool_id:
                    for tool in result["tool_messages"]:
                        if tool.get("id") == tool_id:
                            tool["output"] = msg.content
                            break
                else:
                    result["tool_messages"].append({
                        "name": getattr(msg, "name", "unknown_tool"),
                        "content": getattr(msg, "content", "No content"),
                        "id": getattr(msg, "id", "")
                    })
        
        # Clean up the thinking process
        result["thinking_process"] = result["thinking_process"].strip()
        
        if result["all_text_contents"]:
            result["response_text"] = result["all_text_contents"][-1]
        
        return result

    def _extract_token_usage_and_cost(self, response):
        """Extract token usage information from the response."""
        token_usage = {"input": 0, "output": 0, "total": 0}
        token_cost = {"input": 0, "output": 0, "total": 0}

        if isinstance(response, dict) and "messages" in response:
            for msg in response["messages"]:
                if hasattr(msg, "usage_metadata"):
                    token_usage["input"] += msg.usage_metadata.get("input_tokens", 0)
                    token_usage["output"] += msg.usage_metadata.get("output_tokens", 0)
                    token_usage["total"] += msg.usage_metadata.get("total_tokens", 0)

                    token_cost["input"] += (
                        token_usage["input"]
                        / 1000
                        * MODEL_COST_PER_1K_INPUT_TOKENS.get(self.model.model, 0)
                    )
                    token_cost["output"] += (
                        token_usage["output"]
                        / 1000
                        * MODEL_COST_PER_1K_OUTPUT_TOKENS.get(self.model.model, 0)
                    )
                    token_cost["total"] += token_cost["input"] + token_cost["output"]

        return token_usage, token_cost


    def _extract_visualization_info(self, response_text):
        """Extract visualization information from the response text."""
        # Look for the visualization tag
        viz_match = re.search(r"\[VISUALIZATION:(.*?)\]", response_text)
        if viz_match:
            # Extract the filename or other info
            viz_info = viz_match.group(1).strip()

            # Remove the tag from the response
            clean_response = re.sub(r"\[VISUALIZATION:.*?\]", "", response_text).strip()

            return {
                "should_display": True,
                "viz_info": viz_info,
                "clean_response": clean_response,
            }

        return {
            "should_display": False,
            "viz_info": None,
            "clean_response": response_text,
        }
