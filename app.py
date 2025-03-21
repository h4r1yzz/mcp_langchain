import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="LAS File Chat Assistant",
    initial_sidebar_state="expanded"
)

# Page title
st.title("Chat Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_path" not in st.session_state:
    st.session_state.file_path = None

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# File Upload Section in sidebar
with st.sidebar:
    st.header("File Upload")
    
    uploaded_file = st.file_uploader(
        "Drag and drop files here",
        type=["las"],
        accept_multiple_files=False,
    )
    
    # Handle file upload
    if uploaded_file and (not st.session_state.file_path or uploaded_file.name not in st.session_state.file_path):
        # Upload file to API
        try:
            response = requests.post("http://localhost:8000/upload", files={"file": uploaded_file.getbuffer()})
            
            if response.status_code == 200:
                file_path = response.json()["file_path"]
                st.session_state.file_path = file_path
                st.session_state.uploaded_files[uploaded_file.name] = {
                    "name": uploaded_file.name,
                    "path": file_path
                }
                st.success(f"File uploaded: {uploaded_file.name}")
            else:
                st.error(f"Failed to upload file: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API server. Make sure it's running with: python client_server.py --api")
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.subheader("Uploaded Files")
        for file_name, file_info in st.session_state.uploaded_files.items():
            st.write(f"📄 {file_name}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat Section
st.subheader("Chat here")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about the well log data..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Check if file is uploaded
    if not st.session_state.file_path:
        with st.chat_message("assistant"):
            st.markdown("Please upload a LAS file first.")
        st.session_state.messages.append({"role": "assistant", "content": "Please upload a LAS file first."})
    else:
        # Process with API
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    data = {
                        "file_path": st.session_state.file_path,
                        "query": prompt
                    }
                    response = requests.post("http://localhost:8000/query", json=data)
                    
                    if response.status_code == 200:
                        ai_response = response.json()["response"]
                    else:
                        ai_response = f"Sorry, I encountered an error processing your request. Status code: {response.status_code}"
                except requests.exceptions.ConnectionError:
                    ai_response = "Could not connect to the API server. Make sure it's running with: python client_server.py --api"
                
                # Display the response
                st.markdown(ai_response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

# Add instructions at the bottom
st.caption("To use this app, first start the API server with: `python client_server.py --api`")
