import os
import time


class LASChatController:
    def __init__(self, model, state_manager):
        self.model = model
        self.state = state_manager
        self._initialized = False

    async def initialize(self):
        """Initialize the controller and model."""
        if not self._initialized:
            # Initialize the model
            await self.model.initialize()
            
            # Register any existing file paths
            file_path = self.state.get_file_path()
            if file_path:
                self.model.register_file_path(file_path)
            
            self._initialized = True
            return True
        return False

    async def handle_query(self, query):
        """Handle a user query and update the state."""
        query_time = time.time()

        # Ensure the controller is initialized
        if not self._initialized:
            await self.initialize()

        # Check if file is upload
        if not self.state.get_file_path():
            return {"status": "error", "message": "Please upload a LAS file first."}

        # Process the query
        result = await self.model.process_query(self.state.get_file_path(), query)

        # Check if there was an error
        if result.get("status") == "error":
            return {
                "status": "error",
                "message": result.get("message", "An error occurred while processing the query."),
                "thinking_process": "",
                "tool_messages": [],
                "token_usage": {"input": 0, "output": 0, "total": 0},
                "token_cost": {"input": 0, "output": 0, "total": 0},
                "visualization": None,
                "should_display_viz": False,
            }

        # Update state with results
        self.state.set_thinking_process(result["thinking_process"])
        self.state.set_token_usage(result["token_usage"])
        self.state.set_tool_messages(result["tool_messages"])

        should_display_viz = result["should_display_viz"]

        recent_visualization = None
        if should_display_viz:
            recent_visualization = self._find_recent_visualization(query_time)

        return {
            "status": "success",
            "response_text": result["response_text"],
            "thinking_process": result["thinking_process"],
            "tool_messages": result["tool_messages"],
            "token_usage": result["token_usage"],
            "token_cost": result["token_cost"],
            "visualization": recent_visualization,
            "should_display_viz": should_display_viz,
            "all_text_contents": result.get("all_text_contents", []),
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
        
        # Register the file path with the model
        self.model.register_file_path(file_path)

        return {
            "status": "success",
            "file_path": file_path,
            "file_name": uploaded_file.name,
        }
    
    async def cleanup(self):
        """Clean up resources when the controller is no longer needed."""
        if self._initialized:
            await self.model.cleanup()
            self._initialized = False
            return True
        return False

    def _find_recent_visualization(self, query_time):
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

        if recent_images:
            latest_image = recent_images[0]
            image_path = os.path.join(viz_dir, latest_image)
            viz_type = latest_image.split("_")[0]

            return {"path": image_path, "filename": latest_image, "type": viz_type}

        return None
