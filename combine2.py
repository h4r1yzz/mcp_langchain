import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", verbose=True)
python_path = sys.executable

async def get_story_and_search_results(topic):
    async with MultiServerMCPClient() as client:
        # Connect to the storywriter server
        await client.connect_to_server(
            "storywriter",
            command=python_path,
            args=["write_blog.py"],
            encoding_error_handler="ignore",
        )

        # Connect to the googlesearch server
        await client.connect_to_server(
            "googlesearch",
            command=python_path,
            args=["google_search.py"],
            encoding_error_handler="ignore",
        )

        agent = create_react_agent(model, client.get_tools(), debug=True)
        review_requested = await agent.ainvoke(debug=True, input={"messages": f"Write a story about {topic} and also search in google for it"})
        
        last_tool_message, last_ai_message = get_last_messages(review_requested)
        return last_tool_message, last_ai_message

def get_last_messages(data):
    messages = dict(data).get('messages', [])
    last_tool_message = None
    last_ai_message = None

    for message in messages:
        if isinstance(message, ToolMessage) and message.name == 'search_google':
            last_tool_message = message.content
        elif isinstance(message, ToolMessage) and message.name == 'write_story':
            last_ai_message = message.content

    return last_tool_message, last_ai_message