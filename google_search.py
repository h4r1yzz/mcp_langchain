import os
import asyncio
from mcp.server.fastmcp import FastMCP, Context
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from googlesearch import search  
from typing import Dict, List, Union
from langchain.schema import AIMessage, HumanMessage

load_dotenv(override=True)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

model = ChatAnthropic(api_key=anthropic_api_key, model="claude-3-5-sonnet-20241022", verbose=True)
mcp = FastMCP("googlesearch")

@mcp.tool()
async def search_google(query: str) -> str:
    try:
        search_results = list(search(query, num_results=5))
        if not search_results:
            return "No results found."

        markdown_results = ""
        for idx, result in enumerate(search_results, 1):
            markdown_results += f"**{idx}.** [{result}](<{result}>)\n"
        return markdown_results
    except Exception as e:
        return f"An error occurred while searching Google: {e}"

if __name__ == "__main__":
    mcp.run()