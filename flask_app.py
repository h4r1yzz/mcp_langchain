import os
import sys
import tempfile
import atexit
import json
import asyncio
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from controllers import LASChatController
from models import LASAnalyzerModel
from state import StateManager

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Create temp directory for file uploads
temp_dir = tempfile.mkdtemp()
visualizations_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
os.makedirs(visualizations_dir, exist_ok=True)

# Initialize Anthropic model
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize our components
model_instance = ChatAnthropic(
    api_key=anthropic_api_key,
    model="claude-3-5-haiku-20241022",
    verbose=True,
    max_tokens=8192
)

python_path = sys.executable
las_model = LASAnalyzerModel(model_instance, python_path)
app_state = {}
state_manager = StateManager(app_state)
controller = LASChatController(las_model, state_manager)

# Clean up upon exit the app
atexit.register(lambda: controller.cleanup_session_files())

@app.route('/')
def index():
    """Render the main page."""
    messages = state_manager.get_messages()
    uploaded_files = state_manager.get_uploaded_files()
    return render_template('index.html',
                          messages=messages,
                          uploaded_files=uploaded_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})

    if not file.filename.endswith('.las'):
        return jsonify({"status": "error", "message": "Only LAS files are supported"})

    # Use controller to handle upload
    result = controller.handle_file_upload(file, temp_dir)

    return jsonify(result)

@app.route('/query', methods=['POST'])
def process_query():
    """Process a user query."""
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"status": "error", "message": "No query provided"})

    # Add user message to history
    state_manager.add_message("user", query)

    # Check if file is uploaded
    if not state_manager.get_file_path():
        state_manager.add_message("assistant", "Please upload a LAS file first.")
        return jsonify({
            "status": "error",
            "message": "Please upload a LAS file first.",
            "messages": state_manager.get_messages()
        })

    # Process query using controller
    result = controller.handle_query(query)

    # Add assistant response to history
    state_manager.add_message("assistant", result["response"])

    # Handle Plotly visualizations
    plotly_visualizations = []
    if result.get("plotly_visualizations"):
        for viz in result["plotly_visualizations"]:
            # Read the JSON file containing the Plotly data
            if 'plot_json_path' in viz:
                try:
                    # Check if the file exists
                    if not os.path.exists(viz['plot_json_path']):
                        continue

                    with open(viz['plot_json_path'], 'r') as f:
                        plot_json = json.load(f)

                    # Validate the JSON structure
                    if "plot_data" not in plot_json or "plot_layout" not in plot_json:
                        continue

                    plotly_visualizations.append({
                        "plot_id": viz["plot_id"],
                        "plot_data": plot_json["plot_data"],
                        "plot_layout": plot_json["plot_layout"],
                        "visualization_type": viz.get("visualization_type", ""),
                        "type": "plotly"
                    })
                except Exception:
                    continue
            else:
                continue

    return jsonify({
        "status": "success",
        "response": result["response"],
        "thinking_process": result["thinking_process"],
        "token_usage": result["token_usage"],
        "token_cost": result["token_cost"],
        "tool_messages": result["tool_messages"],
        "plotly_visualizations": plotly_visualizations
    })

@app.route('/visualization/<filename>')
def visualization(filename):
    """Serve visualization images."""
    return send_from_directory(visualizations_dir, filename)

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/clear', methods=['POST'])
def clear_chat():
    controller.cleanup_session_files()
    state_manager.clear_all()
    return jsonify({"status": "success", "message": "Chat and files cleared"})

@app.route('/query_stream')
def process_query_stream():
    query = request.args.get("query", "")

    # Add user message to history
    state_manager.add_message("user", query)

    if not state_manager.get_file_path():
        def early_exit():
            yield "data: " + json.dumps({"status": "error", "message": "Please upload a LAS file first."}) + "\n\n"
        return Response(
            stream_with_context(early_exit()),
            mimetype='text/event-stream'
        )

    async def generate_async():
        response_text = ""
        # Send start event
        yield "data: " + json.dumps({"status": "start"}) + "\n\n"

        try:
            # First, stream the text response
            async for chunk in controller.handle_query_stream(query):
                # Log the chunk for debugging
                print(f"Streaming chunk: {chunk}")

                if "chunk" in chunk:
                    response_text += chunk["chunk"]
                    # Make sure to properly format the SSE data
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif "status" in chunk:
                    # Handle tool messages
                    if chunk["status"] == "tool_messages":
                        # Save tool messages to state
                        if "tool_messages" in chunk and chunk["tool_messages"]:
                            state_manager.set_tool_messages(chunk["tool_messages"])
                            print(f"Updated state with {len(chunk['tool_messages'])} tool messages")

                        # Extract visualizations directly from tool messages
                        plotly_visualizations = []
                        for msg in chunk["tool_messages"]:
                            if msg.get("name") == "visualize_well_log":
                                content = msg.get("content", "")
                                if not content or not isinstance(content, str):
                                    continue

                                try:
                                    # Parse the JSON content
                                    content_json = json.loads(content)

                                    # Extract the visualization metadata
                                    if "plot_id" in content_json and "plot_json_path" in content_json:
                                        # Check if the file exists
                                        if not os.path.exists(content_json['plot_json_path']):
                                            continue

                                        with open(content_json['plot_json_path'], 'r') as f:
                                            plot_json = json.load(f)

                                        # Validate the JSON structure
                                        if "plot_data" not in plot_json or "plot_layout" not in plot_json:
                                            continue

                                        plotly_visualizations.append({
                                            "plot_id": content_json["plot_id"],
                                            "plot_data": plot_json["plot_data"],
                                            "plot_layout": plot_json["plot_layout"],
                                            "visualization_type": content_json.get("visualization_type", ""),
                                            "type": "plotly"
                                        })
                                except json.JSONDecodeError:
                                    # Skip invalid JSON
                                    continue

                        # Send visualizations as a separate event if there are any
                        if plotly_visualizations:
                            viz_data = {
                                "status": "visualizations",
                                "plotly_visualizations": plotly_visualizations
                            }
                            yield f"data: {json.dumps(viz_data)}\n\n"

                        # Pass through the original tool_messages event
                        yield f"data: {json.dumps(chunk)}\n\n"
                    else:
                        # Pass through other status messages
                        yield f"data: {json.dumps(chunk)}\n\n"
                        # If it's a complete status, we're done
                        if chunk["status"] == "complete":
                            # Update state with debug information if available
                            if "thinking_process" in chunk:
                                state_manager.set_thinking_process(chunk["thinking_process"])
                            if "token_usage" in chunk:
                                state_manager.set_token_usage(chunk["token_usage"])
                            if "tool_messages" in chunk:
                                state_manager.set_tool_messages(chunk["tool_messages"])
                            break
                        # If it's an error, we're done
                        if chunk["status"] == "error":
                            break

            # Only add the message to history if we got a response
            if response_text:
                state_manager.add_message("assistant", response_text)
                
            complete_data = {
                "status": "complete",
                "response": response_text,
                "thinking_process": state_manager.get_thinking_process(),
                "token_usage": state_manager.get_token_usage(),
                "token_cost": {"input": 0, "output": 0, "total": 0},  # Default values
                "tool_messages": state_manager.get_tool_messages()
            }
            yield "data: " + json.dumps(complete_data) + "\n\n"
        except Exception as e:
            # Log the error
            print(f"Error in generate_async: {str(e)}")
            # Send an error event
            yield "data: " + json.dumps({"status": "error", "message": f"Server error: {str(e)}"}) + "\n\n"

    def generate_sync():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        gen = generate_async()
        try:
            while True:
                yield loop.run_until_complete(gen.__anext__())
        except StopAsyncIteration:
            pass

    return Response(
        stream_with_context(generate_sync()),
        mimetype='text/event-stream'
    )

if __name__ == '__main__':
    # Disable auto-reloader to prevent conflicts with MCP server
    app.run(debug=True, use_reloader=False, port=5004)