from mcp.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
model = ChatOpenAI(model="gpt-4o-mini", verbose=True)
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

if __name__ == "__main__":
  mcp.run()

