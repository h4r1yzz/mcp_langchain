import re

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
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

    async def process_query(self, file_path, query):
        """Process a query about a LAS file using the LangChain agent and MCP tools."""
        async with MultiServerMCPClient() as client:
            await client.connect_to_server(
                "LAS File Analyzer",
                command=self.python_path,
                args=["file.py"],
                encoding_error_handler="ignore",
            )

            # Create the agent with debug enabled
            agent = create_react_agent(self.model, client.get_tools(), debug=True)

            # Prepare the query
            full_query = f"Using the LAS file at {file_path}, {query}"

            # Create a system message instructing the agent to use a visualization tag
            system_message = """
            When responding to queries about LAS files, if you create a visualization,
            explicitly indicate this in your response with a special tag: [VISUALIZATION:filename].
            Only include this tag if you've actually created a visualization.
            """

            # Process the query with the system message
            response = await agent.ainvoke(
                debug=True,
                input={
                    "messages": [
                        SystemMessage(content=system_message),
                        HumanMessage(content=full_query),
                    ]
                },
            )

            thinking_process, tool_messages = self.extract_ai_and_tool_messages(response)

            # Extract token usage and cost
            token_usage, token_cost = self._extract_token_usage_and_cost(response)

            # Extract the response text
            response_text = self._extract_response_text(response)

            # Extract visualization information
            viz_info = self._extract_visualization_info(response_text)

            return {
                "response_text": viz_info["clean_response"],
                "thinking_process": thinking_process,
                "tool_messages": tool_messages,
                "token_usage": token_usage,
                "token_cost": token_cost,
                "raw_response": response,
                "should_display_viz": viz_info["should_display"],
                "viz_info": viz_info["viz_info"],
            }
    
    def extract_ai_and_tool_messages(self, response):
        if not isinstance(response, dict) or "messages" not in response:
            return "", []

        ai_text_parts = []
        tool_messages = []
        messages = response["messages"]
        total = len(messages)

        for i, msg in enumerate(messages):
            skip_ai = (i == total - 1 and isinstance(msg, AIMessage))
            
            if isinstance(msg, AIMessage) and not skip_ai:
                content = msg.content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") in ("thinking", "text"):
                            ai_text_parts.append(item.get("text") or item.get("thinking", ""))
                        elif isinstance(item, str):
                            ai_text_parts.append(item)
                elif isinstance(content, str):
                    ai_text_parts.append(content)
            
            if isinstance(msg, ToolMessage):
                tool_messages.append({
                    "name": getattr(msg, "name", "unknown_tool"),
                    "content": getattr(msg, "content", "No content"),
                    "id": getattr(msg, "id", "")
                })


        return "\n\n".join(ai_text_parts), tool_messages

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

    def _extract_response_text(self, response):
        """Extract the response text from the response."""
        if isinstance(response, dict) and "messages" in response:
            ai_messages = [
                msg.content
                for msg in response["messages"]
                if isinstance(msg, AIMessage)
            ]
            if ai_messages:
                return ai_messages[-1]

        return "I couldn't process that request."

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