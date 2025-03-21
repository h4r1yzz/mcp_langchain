import asyncio
import sys
import os
import tempfile
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory for file uploads
temp_dir = tempfile.mkdtemp()

# Initialize Anthropic model
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

model = ChatAnthropic(
    api_key=anthropic_api_key, 
    model="claude-3-7-sonnet-20250219", 
    verbose=True, 
    thinking={"type": "enabled", "budget_tokens": 16000}, 
    max_tokens=20000
)

python_path = sys.executable

def parse_ai_messages(response):
    """Extract AI messages from the agent response."""
    if isinstance(response, dict) and "messages" in response:
        return [msg.content for msg in response["messages"] if isinstance(msg, AIMessage)]
    return [response]

# FastAPI models
class QueryRequest(BaseModel):
    file_path: str
    query: str

# FastAPI endpoints
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a LAS file and return its path"""
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"file_path": file_path}

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a query about a LAS file"""
    file_path = request.file_path
    query = request.query
    
    async with MultiServerMCPClient() as client:
        await client.connect_to_server(
            "LAS File Analyzer",
            command=python_path,
            args=["file.py"],
            encoding_error_handler="ignore",
        )
        
        # Create the agent
        agent = create_react_agent(
            model, 
            client.get_tools(), 
            debug=True
        )
        
        # Process the query with better error handling
        try:
            full_query = f"Using the LAS file at {file_path}, {query}"
            response = await agent.ainvoke(debug=False, input={"messages": [HumanMessage(content=full_query)]})
        except Exception as e:
            print(f"Error during agent invocation: {str(e)}")
            return {"response": f"Error processing your query: {str(e)}. Please try a simpler question or check the file format."}
        
        # Extract and return AI response
        ai_messages = parse_ai_messages(response)
        if ai_messages:
            return {"response": ai_messages[-1]}
        
        return {"response": "I couldn't process that request."}

# async def main():
    # print("LAS File Analyzer Interactive Client")
    # print("===================================")
    
    # # Get the file path from the user or use default
    # file_path = input("Enter the path to the LAS file (default: ../1046506109.las): ").strip()
    # if not file_path:
    #     file_path = "../1046506109.las"
    
    # print(f"Using LAS file: {file_path}")
    # print("Connecting to LAS File Analyzer server...")
    
    # async with MultiServerMCPClient() as client:
    #     await client.connect_to_server(
    #         "LAS File Analyzer",
    #         command=python_path,
    #         args=["file.py"],
    #         encoding_error_handler="ignore",
    #     )
        
    #     print("Server connected successfully!")
        
    #     # Create the agent
    #     agent = create_react_agent(model, client.get_tools(), debug=True)
        
    #     # First, analyze the file to get metadata
    #     print("\nAnalyzing LAS file...")
    #     initial_query = f"Analyze the LAS file at {file_path} and summarize its key information."
    #     response = await agent.ainvoke(debug=True, input={"messages": [HumanMessage(content=initial_query)]})
        
    #     # Print the response
    #     ai_messages = parse_ai_messages(response)
    #     for message in ai_messages:
    #         print(message)
        
    #     # Interactive question loop
    #     print("\n\nYou can now ask questions about the LAS file data.")
    #     print("Examples:")
    #     print("  - What is the 7th value of CGXT?")
    #     print("  - What is the value of GRGC at depth 437.358?")
    #     print("  - Show me the value of NPRL at index 100")
    #     print("Type 'exit' or 'quit' to end the session.")
        
    #     # Keep conversation history
    #     conversation = [
    #         HumanMessage(content=initial_query),
    #         AIMessage(content=ai_messages[-1] if ai_messages else "Analysis complete.")
    #     ]
        
    #     while True:
    #         # Get user question
    #         user_input = input("\nYour question: ").strip()
            
    #         # Check if user wants to exit
    #         if user_input.lower() in ['exit', 'quit', 'q']:
    #             print("Exiting. Goodbye!")
    #             break
            
    #         if not user_input:
    #             continue
            
    #         # Add user message to conversation
    #         conversation.append(HumanMessage(content=user_input))
            
    #         # Get response from agent
    #         print("Thinking...")
    #         response = await agent.ainvoke(debug=False, input={"messages": conversation})
            
    #         # Extract and print AI response
    #         ai_messages = parse_ai_messages(response)
    #         for message in ai_messages:
    #             print("\nAnswer:", message)
            
    #         # Add AI response to conversation
    #         if ai_messages:
    #             conversation.append(AIMessage(content=ai_messages[-1]))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
