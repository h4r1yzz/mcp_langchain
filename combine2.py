import streamlit as st
import asyncio
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(override=True)

# Initialize the Anthropic model
model = ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022", verbose=True)

# Get the current Python executable path
python_path = sys.executable

# Streamlit app title
st.title("AI Agent Orchestrator")
st.write("This app connects the Story Writer AI and Google Search AI to perform tasks like writing stories and searching the web.")

# Input for the user's task
task = st.text_area("Enter your task:", "Write a story about Lord Krishna and Arjuna, search for related information on Google, and generate images for it.")

async def run_agent(task):
    """Run the agent with the given task."""
    async with MultiServerMCPClient() as client:
        # Connect to the Story Writer AI server
        await client.connect_to_server(
            "storywriter",
            command=python_path,
            args=["story_writer.py"],  # Path to your Story Writer AI script
            encoding_error_handler="ignore",
        )

        # Connect to the Google Search AI server
        await client.connect_to_server(
            "googlesearch",
            command=python_path,
            args=["google_search.py"],  # Path to your Google Search AI script
            encoding_error_handler="ignore",
        )

        # Create a ReAct agent with the connected tools
        agent = create_react_agent(model, client.get_tools(), debug=True)

        # Invoke the agent to perform the task
        result = await agent.ainvoke(
            debug=True,
            input={"messages": task}
        )

        # Parse and return the AI responses
        return parse_ai_messages(result)

def parse_ai_messages(data):
    """Parse and format AI messages from the agent's response."""
    messages = dict(data).get('messages', [])
    formatted_ai_responses = []

    for message in messages:
        if isinstance(message, AIMessage):
            formatted_message = f"### AI Response:\n\n{message.content}\n\n"
            formatted_ai_responses.append(formatted_message)

    return formatted_ai_responses

# Button to execute the task
if st.button("Run Task"):
    with st.spinner("Executing task..."):
        # Run the agent asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_agent(task))
        loop.close()

        # Display the results
        st.write("## Task Results")
        for response in result:
            st.markdown(response)

        st.success("Task completed successfully!")