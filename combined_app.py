import asyncio
import atexit
import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Import our MVC components
from controllers import LASChatController
from models import LASAnalyzerModel
from state import StateManager

# Load environment variables
load_dotenv(override=True)

# Set page configuration
st.set_page_config(
    page_title="LAS File Chat Assistant", initial_sidebar_state="expanded"
)

# Page title
st.title("Chat Assistant")

# Create temp directory for file uploads
temp_dir = tempfile.mkdtemp()

# Initialize Anthropic model
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    st.error(
        "ANTHROPIC_API_KEY not found in environment variables. Please set it in the .env file."
    )
    st.stop()

# Initialize our components
model_instance = ChatAnthropic(
    api_key=anthropic_api_key,
    model="claude-3-7-sonnet-20250219",
    verbose=True,
    thinking={"type": "enabled", "budget_tokens": 16000},
    max_tokens=20000,
)

python_path = sys.executable
las_model = LASAnalyzerModel(model_instance, python_path)
state_manager = StateManager(st.session_state)
controller = LASChatController(las_model, state_manager)

# Initialize the controller
if "controller_initialized" not in st.session_state:
    st.session_state.controller_initialized = False

# Initialize the controller if not already initialized
if not st.session_state.controller_initialized:
    asyncio.run(controller.initialize())
    st.session_state.controller_initialized = True

# Register cleanup function to be called on exit
def cleanup_resources():
    if st.session_state.controller_initialized:
        # Create a new event loop for cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(controller.cleanup())
        loop.close()
        st.session_state.controller_initialized = False

# Register the cleanup function
atexit.register(cleanup_resources)

# File Upload Section in sidebar
with st.sidebar:
    st.header("File Upload")

    uploaded_file = st.file_uploader(
        "Drag and drop files here",
        type=["las"],
        accept_multiple_files=False,
    )

    # Handle file upload
    if uploaded_file and (
        not state_manager.get_file_path()
        or uploaded_file.name not in state_manager.get_file_path()
    ):
        # Use controller to handle upload
        result = controller.handle_file_upload(uploaded_file, temp_dir)
        if result["status"] == "success":
            st.success(f"File uploaded: {result['file_name']}")

    # Display uploaded files
    uploaded_files = state_manager.get_uploaded_files()
    if uploaded_files:
        st.subheader("Uploaded Files")
        for file_name, file_info in uploaded_files.items():
            st.write(f"📄 {file_name}")

    # Clear chat button
    if st.button("Clear Chat"):
        state_manager.clear_all()
        st.rerun()

# Display chat history
for message in state_manager.get_messages():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about the well log data..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to history
    state_manager.add_message("user", prompt)

    # Check if file is uploaded
    if not state_manager.get_file_path():
        with st.chat_message("assistant"):
            st.markdown("Please upload a LAS file first.")
        state_manager.add_message("assistant", "Please upload a LAS file first.")
    else:
        # Process query using controller
        with st.spinner("Processing query..."):
            result = asyncio.run(controller.handle_query(prompt))

        with st.expander("Thinking, Tools & Token Usage"):
            # Create tabs for different sections
            tab1, tab2, tab3 = st.tabs(["Token Usage", "Thinking Process", "Tool Messages"])
            
            with tab1:
                # Display token usage
                st.subheader("Token Usage and Cost")
                token_usage = result["token_usage"]
                token_cost = result["token_cost"]
                st.text(
                    f"Token in / out / total: {token_usage['input']} / {token_usage['output']} / {token_usage['total']}"
                )
                st.text(
                    f"Cost in / out / total: ${token_cost['input']:.2f} / ${token_cost['output']:.2f} / ${token_cost['total']:.2f}"
                )
            
            with tab2:
                # Display thinking process and text content
                st.subheader("Thinking Process")
                if result["thinking_process"]:
                    st.text(result["thinking_process"])
                else:
                    st.info(
                        "No thinking process available yet. Ask a question to see the agent's reasoning."
                    )
                
                # Display all text contents
                st.subheader("Text Content")
                if "all_text_contents" in result and result["all_text_contents"]:
                    # Display each text content with a separator
                    for i, text in enumerate(result["all_text_contents"]):
                        if i > 0:
                            st.divider()
                        st.text(text)
                elif result["response_text"]:
                    # Fallback to response_text if all_text_contents is not available
                    st.text(result["response_text"])
                else:
                    st.info(
                        "No text content available yet."
                    )
            
            with tab3:
                # Display tool messages
                if result.get("tool_messages") and len(result["tool_messages"]) > 0:
                    for i, tool_msg in enumerate(result["tool_messages"]):
                        st.subheader(f"Tool: {tool_msg.get('name', 'Unknown')} ({i+1}/{len(result['tool_messages'])})")
                        st.json(tool_msg)
                        if i < len(result["tool_messages"]) - 1:
                            st.divider()
                else:
                    st.info("No tool messages available yet. Ask a question that requires tool use.")

        # Display visualization if needed
        if result["should_display_viz"] and result["visualization"]:
            viz = result["visualization"]
            st.image(viz["path"])

        # Now display the assistant's response
        with st.chat_message("assistant"):
            # Display the processed response
            st.markdown(result["response_text"])

        # Add assistant response to history
        state_manager.add_message("assistant", result["response_text"])
