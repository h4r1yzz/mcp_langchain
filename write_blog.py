import os
import asyncio
from mcp.server.fastmcp import FastMCP, Context
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from typing import Dict, List, Union
from langchain.schema import AIMessage, HumanMessage

# Load environment variables
load_dotenv(override=True)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize the Anthropic model
model = ChatAnthropic(api_key=anthropic_api_key, model="claude-3-5-sonnet-20241022", verbose=True)

# Initialize FastMCP
mcp = FastMCP("storywriter")

@mcp.tool()
async def write_story(topic: str) -> str:
    """Write a story.
    Args:
        topic: The story topic  
    Returns:
        The written story as a string
    """
    try:
        messages = [
            (
                "system", 
                "You are a talented story writer. Create an engaging short story on the given topic in a maximum of 100 words. Provide the output in markdown format only.",
            ),
            ("human", f"The topic is: {topic}"),
        ]
        ai_msg = await model.ainvoke(messages)
        return ai_msg.content
    except Exception as e:
        return f"An error occurred while writing story: {e}"

# Run the FastMCP tool and test the function
if __name__ == "__main__":
    mcp.run()