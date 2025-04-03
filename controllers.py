import os
import time

class LASChatController:
    def __init__(self, model, state_manager):
        self.model = model
        self.state = state_manager

    def handle_query(self, query):
        """Handle a user query and update the state."""
        query_time = time.time()

        # Check if file is uploaded
        if not self.state.get_file_path():
            return {"status": "error", "message": "Please upload a LAS file first."}

        chat_history = self.state.session_state.messages
        
        result = self.model.process_query(
            self.state.get_file_path(), 
            query,
            chat_history
        )

        # Update state with results
        self.state.set_thinking_process(result["thinking_process"])
        self.state.set_token_usage(result["token_usage"])
        self.state.set_tool_messages(result["tool_messages"])

        should_display_viz = result["should_display_viz"]

        visualizations = []
        if should_display_viz:
            visualizations = self._find_recent_visualizations(query_time)

        return {
            "status": "success",
            "response": result["response_text"],
            "thinking_process": result["thinking_process"],
            "tool_messages": result["tool_messages"],
            "token_usage": result["token_usage"],
            "token_cost": result["token_cost"],
            "visualizations": visualizations,
            "should_display_viz": should_display_viz,
        }

    def handle_file_upload(self, uploaded_file, temp_dir):
        """Handle file upload and update state."""
        if not uploaded_file:
            return {"status": "error", "message": "No file provided"}

        # Save the uploaded file to the temp directory
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Update state
        self.state.set_file_path(file_path)
        self.state.add_uploaded_file(uploaded_file.name, file_path)

        return {
            "status": "success",
            "file_path": file_path,
            "file_name": uploaded_file.name,
        }

    def _find_recent_visualizations(self, query_time):
        """Find the most recent visualization created after query_time."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        viz_dir = os.path.join(current_dir, "visualizations")

        if not os.path.exists(viz_dir):
            return None

        # Get all image files in the directory
        image_files = [f for f in os.listdir(viz_dir) if f.endswith(".png")]

        # Sort by creation time (most recent first)
        image_files.sort(
            key=lambda x: os.path.getctime(os.path.join(viz_dir, x)), reverse=True
        )

        # Find images created after query_time
        recent_images = [
            f
            for f in image_files
            if os.path.getctime(os.path.join(viz_dir, f)) >= query_time
        ]

        visualizations = []
        for image_file in recent_images:
            image_path = os.path.join(viz_dir, image_file)
            viz_type = image_file.split("_")[0]
            visualizations.append({
                "path": image_path, 
                "filename": image_file, 
                "type": viz_type
            })

        return visualizations
