import re
import asyncio

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
        self.model: ChatAnthropic = model
        self.python_path = python_path
        self.agent = None

    def process_query(self, file_path, query, chat_history=None):
        if chat_history is None:
            chat_history = []
        return asyncio.run(self._process_query_async(file_path, query, chat_history))

    async def _process_query_async(self, file_path, query, chat_history):
        async with MultiServerMCPClient() as client:
            await client.connect_to_server(
                "LAS File Analyzer",
                command=self.python_path,
                args=["file.py"],
                encoding_error_handler="ignore",
            )

            self.agent = create_react_agent(self.model, client.get_tools(), debug=True)

            if isinstance(file_paths, str):
                file_paths = [file_paths]

            # Prepare the query with information about all files
            if len(file_paths) == 1:
                file_info = f"Using the LAS file at {file_paths[0]}"
            else:
                files_list = "\n".join([f"- {path}" for path in file_paths])
                file_info = f"Using the following LAS files:\n{files_list}"

            full_query = f"{file_info}, {query}"

            system_message = """
            When responding to queries about LAS files:
            1. If you create a visualization, explicitly indicate this in your response with a special tag: [VISUALIZATION:filename].
            2. Maintain context from previous messages in the conversation.
            3. When the user refers to something mentioned earlier (like "show me that", "yes please do that"),
               understand what they're referring to based on the conversation history.
            4. If you offer to show visualizations or perform analyses, remember these offers when the user
               responds affirmatively without explicitly restating what they want.
            5. You can access LAS file content using the access_resource tool with a resource URI.
               Example: access_resource("las://file/sample.las") where "sample.las" is just the filename (without the path).
               This provides more efficient access to the file content than repeatedly calling the analyzer tool.
               The tool returns a dictionary with the file content and metadata.
            6. When multiple LAS files are provided, you should analyze ALL files by calling las_file_analyzer on EACH file path.
            7. When multiple files are available, organize your response to clearly show information from each file.
            8. For each file, include the filename, well details, and key curve information.
            9. When comparing files, create a structured comparison highlighting similarities and differences.
            10. When asked to compare or analyze porosity or any other measurement, you MUST identify and include ALL relevant curves
                for that measurement type from each well, not just one curve per well.
            11. For neutron porosity specifically, you MUST search for and include ALL curves with the following characteristics:
                - Curves with mnemonics containing: NPOR, NPHI, NPRL, NPRS, NPRD, CNL, TNPH, SPOR, SPHI, SNP, PHIN, TPHI, TNPL
                - Curves with descriptions containing words like "neutron" and "porosity"
                - You MUST include ALL such curves from EACH well in your analysis and visualizations
            12. When creating visualizations for neutron porosity, you MUST include ALL identified neutron porosity curves in the plot with clear labels.
            13. Example: If a well has both NPOR and NPRL curves, you MUST include BOTH in your analysis and visualizations when discussing neutron porosity.
            14. When asked to compare neutron porosity between specific depths, first identify ALL neutron porosity curves in each well, then create
                visualizations that include ALL these curves limited to the specified depth range.
            """

            messages = [SystemMessage(content=system_message)]
            messages.extend(chat_history)
            messages.append(HumanMessage(content=full_query))

            # Process the query with the system message and conversation history
            response = await self.agent.ainvoke(
                debug=True,
                input={"messages": messages}
            )

            thinking_process, tool_messages = self.extract_ai_and_tool_messages(response)
            token_usage, token_cost = self._extract_token_usage_and_cost(response)
            response_text = self._extract_response_text(response)
            viz_info = self._extract_visualization_info(response_text)

            return {
                "response_text": viz_info["clean_response"],
                "thinking_process": thinking_process,
                "tool_messages": tool_messages,
                "token_usage": token_usage,
                "token_cost": token_cost,
                "raw_response": response,
                "should_display_viz": viz_info["should_display_viz"],
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
            # TODO: token cost calculation seems to be different
            # total seems to be cumulative while input and output are not
            for msg in response["messages"]:
                if hasattr(msg, "usage_metadata")and msg.usage_metadata is not None:
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
                last_message = ai_messages[-1]
                
                if isinstance(last_message, list):
                    text_parts = []
                    for item in last_message:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    return "".join(text_parts)
                
                # Handle the case where content is a string
                return last_message

        return "I couldn't process that request."

    def _extract_visualization_info(self, response_text):
        """Extract visualization information from the response text."""
        # Ensure response_text is a string
        if not isinstance(response_text, str):
            try:
                response_text = str(response_text)
            except:
                return {
                    "should_display_viz": False,
                    "viz_info": [],
                    "clean_response": response_text,
                }
        
        # Look for all visualization tags
        viz_matches = re.findall(r"\[VISUALIZATION:(.*?)\]", response_text)
        
        if viz_matches:
            viz_infos = [match.strip() for match in viz_matches]
            clean_response = re.sub(r"\[VISUALIZATION:.*?\]", "", response_text).strip()

            return {
                "should_display_viz": True,
                "viz_info": viz_infos,  
                "clean_response": clean_response,
            }

        return {
            "should_display_viz": False,
            "viz_info": [],
            "clean_response": response_text,
        }