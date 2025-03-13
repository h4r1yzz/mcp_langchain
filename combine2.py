import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatAnthropic(model="claude-3-7-sonnet-20250219", verbose=True, thinking={"type": "enabled", "budget_tokens": 16000}, max_tokens=20000)
python_path = sys.executable

async def get_search_and_chat_results(query: str, files_context: str = "") -> tuple[str, str, str, dict]:
    async with MultiServerMCPClient() as client:
        # Connect to the googlesearch server
        await client.connect_to_server(
            "googlesearch",
            command=python_path,
            args=["google_search.py"],
            encoding_error_handler="ignore",
        )

        # Connect to the llm_chat server
        await client.connect_to_server(
            "llm_chat",
            command=python_path,
            args=["llm.py"],
            encoding_error_handler="ignore",
        )

        agent = create_react_agent(model, client.get_tools(), debug=True)
        
        input_message = query
        if files_context:
            input_message = f"Context from uploaded files:\n{files_context}\n\nUser question: {query}"
        
        review_requested = await agent.ainvoke(debug=True, input={"messages": input_message})
        
        search_results, chat_response, thinking_steps, token_info = get_last_messages(review_requested)
        return search_results or "", chat_response or "I apologize, but I couldn't process your request. Please try again.", thinking_steps, token_info

def get_last_messages(data):
    messages = dict(data).get('messages', [])
    last_search_message = None
    last_chat_message = None
    thinking_steps = None
    token_info = {}

    for message in messages:
        if isinstance(message, ToolMessage) and message.name == 'search_google':
            last_search_message = message.content
        elif isinstance(message, ToolMessage) and message.name == 'chat_with_llm':
            if isinstance(message.content, list):
                last_chat_message = next((item['text'] for item in message.content if item['type'] == 'text'), message.content)
                thinking_steps = next((item['thinking'] for item in message.content if item['type'] == 'thinking'), None)
            else:
                last_chat_message = message.content
        elif isinstance(message, AIMessage):  
            if isinstance(message.content, list):
                last_chat_message = next((item['text'] for item in message.content if item['type'] == 'text'), message.content)
                thinking_steps = next((item['thinking'] for item in message.content if item['type'] == 'thinking'), None)
            else:
                last_chat_message = message.content
            token_info = message.response_metadata['usage']
            token_info.setdefault('total_tokens', token_info.get('input_tokens', 0) + token_info.get('output_tokens', 0))

    return last_search_message, last_chat_message, thinking_steps, token_info