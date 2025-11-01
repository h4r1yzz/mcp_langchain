import asyncio
import json
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class CustomStreamingHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tokens = []
        self.response_text = ""

    def on_llm_new_token(self, token: str, **_) -> None:
        self.tokens.append(token)
        self.response_text += token

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
        self.system_message = """
            When responding to queries about LAS files:
            1. Maintain context from previous messages in the conversation.
            2. When the user refers to something mentioned earlier (like "show me that", "yes please do that"),
               understand what they're referring to based on the conversation history.
            3. If you offer to show visualizations or perform analyses, remember these offers when the user
               responds affirmatively without explicitly restating what they want.
            4. When multiple LAS files are provided, you should analyze ALL files by calling get_file_metadata on EACH file path.
            5. When multiple files are available, organize your response to clearly show information from each file.
            6. For each file, include the filename, well details, and key curve information.
            7. When comparing files, create a structured comparison highlighting similarities and differences.

            IMPORTANT FILE TYPE HANDLING:
            1. Always identify and differentiate between zone label files (containing ZONENAME or ZONE curves) and well log files.
            2. For zone label files, extract zone names and boundaries.
            3. For well log files, extract measurement data.
            4. When both zone label files and well log files are available, correlate zone information with measurement data.
            5. Always specify which depth reference system (MD, TVD, TVDSS) is being used when reporting zone information.
            6. When users ask about zone names or boundaries, always indicate which depth reference system the zones are measured in.
            7. Use the file_type metadata from get_file_metadata to identify zone label files vs well log files.

            IMPORTANT: To avoid token limitations, use the get_ascii_data tool to run Python code on the LAS dataframes instead of requesting all the data.

            When using the get_ascii_data tool:
            1. First call get_file_metadata to get metadata about the file and available curves.
            2. ALWAYS translate the user's natural language query into proper executable Python code.
            3. NEVER pass the user's raw question text directly to get_ascii_data.
            4. Write clear, efficient Python code that uses pandas and numpy operations.
            5. The code will have access to the following variables:
               - df: The pandas DataFrame containing the LAS data (index is depth)
               - las: The lasio object containing the raw LAS file data
               - np: The numpy module
               - pd: The pandas module
            6. For complex queries, break them down into multiple code executions if needed.
            7. Always handle potential errors in your code (e.g., check if columns exist before using them).
            8. Limit the amount of data returned by filtering, aggregating, or sampling when appropriate.

            Examples of translating natural language to code:

            1. Query: "What is the average value of GR?"
               Code: ```
               # Check if GR curve exists
               if 'GR' in df.columns:
                   # Calculate and print the average value
                   avg_gr = df['GR'].mean()
                   print(f"The average value of GR is {avg_gr:.4f}")
               else:
                   print("GR curve not found in the dataset")
               ```

            2. Query: "What is the 2nd non-null value for CGXT?"
               Code: ```
               # Check if CGXT curve exists
               if 'CGXT' in df.columns:
                   # Get non-null values
                   non_null_values = df['CGXT'].dropna()
                   if len(non_null_values) >= 2:
                       # Get the 2nd non-null value
                       second_value = non_null_values.iloc[1]
                       depth = non_null_values.index[1]
                       print(f"The 2nd non-null value for CGXT is {second_value} at depth {depth}")
                   else:
                       print(f"CGXT has only {len(non_null_values)} non-null values, not enough to get the 2nd value")
               else:
                   print("CGXT curve not found in the dataset")
               ```

            3. Query: "Show me depths where resistivity is greater than 100"
               Code: ```
               # Look for resistivity curves
               res_curves = [col for col in df.columns if any(x in col.upper() for x in ['RT', 'RESD', 'RES', 'ILD'])]

               if res_curves:
                   for curve in res_curves:
                       high_res = df[df[curve] > 100]
                       if not high_res.empty:
                           print(f"Curve {curve} exceeds 100 at {len(high_res)} depths")
                           print(f"First 5 depths: {high_res.index[:5].tolist()}")
                       else:
                           print(f"No depths found where {curve} exceeds 100")
               else:
                   print("No resistivity curves found in the dataset")
               ```

            When working with zone information:
            1. Always identify which depth reference system is being used (MD, TVD, TVDSS)
            2. When reporting zone boundaries, specify the depth reference system
            3. For zone analysis code, use patterns like:
               ```
               # Identify depth reference system
               depth_col = df.index.name or df.columns[0]
               depth_system = "MD"  # Default
               for col in df.columns:
                   if "TVD" in col.upper():
                       depth_col = col
                       depth_system = "TVD" if "SS" not in col.upper() else "TVDSS"
                       break
               
               # Report zones with reference system
               print(f"Analyzing zones using {depth_system} reference system")
               ```

            Other useful code patterns:
            - To find the average value of a curve: `df['CURVE_NAME'].mean()`
            - To find values in a depth range: `df.loc[min_depth:max_depth, 'CURVE_NAME']`
            - To find correlations: `df[['CURVE1', 'CURVE2']].corr()`
            - To identify zones where a curve exceeds a threshold: `df[df['CURVE_NAME'] > threshold]`

            When asked to compare or analyze porosity or any other measurement, identify ALL relevant curves
            for that measurement type from each well, not just one curve per well.

            For neutron porosity specifically, search for and include ALL curves with the following characteristics:
            - Curves with mnemonics containing: NPOR, NPHI, NPRL, NPRS, NPRD, CNL, TNPH, SPOR, SPHI, SNP, PHIN, TPHI, TNPL
            - Curves with descriptions containing words like "neutron" and "porosity"

            When creating visualizations, include ALL identified relevant curves in the plot with clear labels.
            """

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

            if isinstance(file_path, str):
                file_paths = [file_path]
            else:
                file_paths = file_path

            # Prepare the query with information about all files
            if len(file_paths) == 1:
                file_info = f"Using the LAS file at {file_paths[0]}"
            else:
                files_list = "\n".join([f"- {path}" for path in file_paths])
                file_info = f"Using the following LAS files:\n{files_list}"

            full_query = f"{file_info}, {query}"

            # Use the system message defined in the class
            system_message = self.system_message

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
            plotly_visualizations = self._extract_plotly_visualizations(tool_messages)

            return {
                "response_text": response_text,
                "thinking_process": thinking_process,
                "tool_messages": tool_messages,
                "token_usage": token_usage,
                "token_cost": token_cost,
                "raw_response": response,
                "plotly_visualizations": plotly_visualizations
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

    def _extract_plotly_visualizations(self, tool_messages):
        visualizations = []

        for msg in tool_messages:
            # Check if this is a visualization tool message
            if isinstance(msg, dict) and msg.get("name") == "get_visualization":
                content = msg.get("content", "")
                if not content or not isinstance(content, str):
                    continue

                try:
                    # Parse the JSON content
                    content_json = json.loads(content)

                    # Extract the visualization metadata
                    if "plot_id" in content_json and "plot_json_path" in content_json:
                        visualizations.append({
                            "plot_id": content_json["plot_id"],
                            "plot_json_path": content_json["plot_json_path"],
                            "visualization_type": content_json.get("visualization_type", ""),
                            "curves": content_json.get("curves", []),
                            "metadata": content_json.get("metadata", {})
                        })
                except json.JSONDecodeError:
                    # Skip invalid JSON
                    continue

        return visualizations

    async def _chunk_and_stream_text(self, text, current_text=""):
        """Helper method to chunk text and stream it."""
        if not text or len(text) <= len(current_text):
            return

        new_content = text[len(current_text):]
        words = new_content.split(' ')

        chunk_size = min(3, len(words))
        for i in range(0, len(words), chunk_size):
            word_chunk = ' '.join(words[i:i+chunk_size])
            if word_chunk:
                yield {"chunk": word_chunk + ('' if i+chunk_size >= len(words) else ' ')}
                await asyncio.sleep(0.05)

    async def _extract_fallback_response(self, response):
        if not isinstance(response, dict) or "messages" not in response:
            return ""

        ai_messages = [
            msg for msg in response["messages"]
            if isinstance(msg, AIMessage)
        ]

        if not ai_messages:
            return ""

        last_message = ai_messages[-1]
        response_text = ""

        if isinstance(last_message.content, str):
            response_text = last_message.content
        elif isinstance(last_message.content, list):
            for item in last_message.content:
                if isinstance(item, dict) and 'text' in item:
                    response_text += item['text']

        return response_text

    async def process_query_stream_async(self, file_paths, query, chat_history=None):
        if chat_history is None:
            chat_history = []

        # Start the streaming response
        yield {"status": "start"}

        try:
            client = MultiServerMCPClient()
            await client.connect_to_server(
                "LAS File Analyzer",
                command=self.python_path,
                args=["file.py"],
                encoding_error_handler="ignore",
            )

            streaming_agent = create_react_agent(self.model, client.get_tools(), debug=True)

            if isinstance(file_paths, str):
                file_paths = [file_paths]

            if len(file_paths) == 1:
                file_info = f"Using the LAS file at {file_paths[0]}"
            else:
                files_list = "\n".join([f"- {path}" for path in file_paths])
                file_info = f"Using the following LAS files:\n{files_list}"

            full_query = f"{file_info}, {query}"

            system_message = self.system_message

            messages = [SystemMessage(content=system_message)]
            messages.extend(chat_history)
            messages.append(HumanMessage(content=full_query))

            streaming_handler = CustomStreamingHandler()
            response_text = ""

            # Start the agent execution with streaming
            agent_task = asyncio.create_task(
                streaming_agent.ainvoke(
                    input={"messages": messages},
                    config={"callbacks": [streaming_handler]}
                )
            )
            # Phase 1: Stream tokens while agent is running
            while not agent_task.done():
                if len(streaming_handler.response_text) > len(response_text):
                    async for chunk in self._chunk_and_stream_text(streaming_handler.response_text, response_text):
                        yield chunk
                    response_text = streaming_handler.response_text
                await asyncio.sleep(0.01)

            try:
                response = await agent_task
                # Phase 2: Stream any final tokens from the handler
                if len(streaming_handler.response_text) > len(response_text):
                    async for chunk in self._chunk_and_stream_text(streaming_handler.response_text, response_text):
                        yield chunk
                    response_text = streaming_handler.response_text
                # Phase 3: Fallback if streaming handler didn't capture anything
                if not response_text:
                    fallback_text = await self._extract_fallback_response(response)
                    if fallback_text:
                        async for chunk in self._chunk_and_stream_text(fallback_text):
                            yield chunk
                        response_text = fallback_text

            except Exception as e:
                print(f"Error in agent execution: {str(e)}")
                yield {"status": "error", "message": f"Error in agent execution: {str(e)}"}
                return  # Exit early on error

            thinking_process, tool_messages = self.extract_ai_and_tool_messages(response)
            token_usage, token_cost = self._extract_token_usage_and_cost(response)

            if tool_messages:
                yield {"status": "tool_messages", "tool_messages": tool_messages}

            yield {
                "status": "complete",
                "response": response_text,
                "thinking_process": thinking_process,
                "token_usage": token_usage,
                "token_cost": token_cost,
                "tool_messages": tool_messages if tool_messages else []
            }

            await client.close()

        except Exception as e:
            print(f"Error during streaming: {str(e)}")
            yield {"status": "error", "message": f"Error during streaming: {str(e)}"}