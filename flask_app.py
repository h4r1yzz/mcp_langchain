import os
import sys
import tempfile
import atexit

from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
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

# find where all directory of them 
print(f"\nTemporary directory: {temp_dir}")
print(f"Visualizations directory: {visualizations_dir}\n")

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

    visualizations = []
    if result["should_display_viz"] and result.get("visualizations"):
        for viz in result["visualizations"]:
            viz_path = viz["path"]
            viz_filename = os.path.basename(viz_path)
            visualizations.append({
                "filename": viz_filename,
                "url": url_for('visualization', filename=viz_filename)
            })

    return jsonify({
        "status": "success",
        "response": result["response"],
        "thinking_process": result["thinking_process"],
        "token_usage": result["token_usage"],
        "token_cost": result["token_cost"],
        "tool_messages": result["tool_messages"],
        "visualizations": visualizations,
        "should_display_viz": result["should_display_viz"]
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

if __name__ == '__main__':
    # Disable auto-reloader to prevent conflicts with MCP server
    app.run(debug=True, use_reloader=False, port=5003)
