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

model = ChatAnthropic(api_key=anthropic_api_key, model="gpt-4o", verbose=True)
mcp = FastMCP("google search")

@mcp.tool()
async def search_google(query: str) -> str:
    """Search Google for the query and return results as markdown formatted text.
    Args:
        query: The search query
    Returns:
        Search results formatted in markdown
    """
    try:
        search_results = list(search(query, num_results=5))  # Limiting to 5 results
        if not search_results:
            return "No results found."

        # Format the search results in Markdown
        markdown_results = "### Search Results:\n\n"
        for idx, result in enumerate(search_results, 1):
            markdown_results += f"**{idx}.** [{result}](<{result}>)\n"
        return markdown_results
    except Exception as e:
        return f"An error occurred while searching Google: {e}"

if __name__ == "__main__":
    mcp.run()