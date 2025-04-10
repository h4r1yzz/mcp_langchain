import os
import time

class LASChatController:
    def __init__(self, model, state_manager):
        self.model = model
        self.state = state_manager

    def handle_query(self, query):
        query_time = time.time()

        # Get all uploaded files
        uploaded_files = self.state.get_uploaded_files()
        if not uploaded_files:
            return {"status": "error", "message": "Please upload a LAS file first."}

        chat_history = self.state.get_messages()
        # Get all file paths from uploaded files
        file_paths = [file_info['path'] for file_info in uploaded_files.values()]

        result = self.model.process_query(
            file_paths,
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
        if not uploaded_file:
            return {"status": "error", "message": "No file provided"}

        # Save the uploaded file to the temp directory
        file_path = os.path.join(temp_dir, uploaded_file.filename)
        with open(file_path, "wb") as f:
        # Change getbuffer to read for compatible with Flask
            f.write(uploaded_file.read())

        # Update state
        self.state.set_file_path(file_path)
        self.state.add_uploaded_file(uploaded_file.filename, file_path)

        return {
            "status": "success",
            "file_path": file_path,
            "file_name": uploaded_file.filename,
        }

    def _find_recent_visualizations(self, query_time):
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

    def cleanup_session_files(self):
        uploaded_files = self.state.get_uploaded_files()

        # Remove each uploaded file
        for file_info in uploaded_files.values():
            file_path = file_info.get('path')
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

        # Also clear all visualization files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        viz_dir = os.path.join(current_dir, "visualizations")
        if os.path.exists(viz_dir):
            for filename in os.listdir(viz_dir):
                file_path = os.path.join(viz_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass