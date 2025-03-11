import os
import asyncio
from mcp.server.fastmcp import FastMCP, Context
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from typing import Dict, List, Union
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv(override=True)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

model = ChatAnthropic(api_key=anthropic_api_key, model="claude-3-sonnet-20240229", verbose=True)
mcp = FastMCP("llm_chat")


@mcp.tool()
async def chat_with_llm(message: str) -> str:
    """Simple chat with the LLM without any memory context."""
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content=message)
    ]
    response = await model.ainvoke(messages)
    return response.content


if __name__ == "__main__":
    mcp.run()
    
    