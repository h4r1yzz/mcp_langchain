import os

class LASChatController:
    def __init__(self, model, state_manager):
        self.model = model
        self.state = state_manager

    def handle_query(self, query):
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

        plotly_visualizations = result.get("plotly_visualizations", [])

        return {
            "status": "success",
            "response": result["response_text"],
            "thinking_process": result["thinking_process"],
            "tool_messages": result["tool_messages"],
            "token_usage": result["token_usage"],
            "token_cost": result["token_cost"],
            "plotly_visualizations": plotly_visualizations,
        }

    def handle_file_upload(self, uploaded_file, temp_dir):
        if not uploaded_file:
            return {"status": "error", "message": "No file provided"}

        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded file to the temp directory
        file_path = os.path.join(temp_dir, uploaded_file.filename)
        with open(file_path, "wb") as f:
        # Change getbuffer to read for compatible with Flask
            f.write(uploaded_file.read())

        # Extract metadata from the LAS file
        file_info = self._extract_las_metadata(file_path)

        # Update state
        self.state.set_file_path(file_path)
        self.state.add_uploaded_file(uploaded_file.filename, file_path, file_info)

        return {
            "status": "success",
            "file_path": file_path,
            "file_name": uploaded_file.filename,
            "file_info": file_info
        }

    async def handle_query_stream(self, query):
        uploaded_files = self.state.get_uploaded_files()
        if not uploaded_files:
            yield {"status": "error", "message": "Please upload a LAS file first."}
            return

        chat_history = self.state.get_messages()
        file_paths = [file_info['path'] for file_info in uploaded_files.values()]

        try:
            async for chunk in self.model.process_query_stream_async(
                file_paths,
                query,
                chat_history  
            ):
                yield chunk 
        except Exception as e:
            yield {"status": "error", "message": f"Error during streaming: {str(e)}"}

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

    def _extract_las_metadata(self, file_path):
        """Extract metadata from a LAS file for display in the UI."""
        try:
            import lasio
            las = lasio.read(file_path)

            metadata = {}

            # Well name
            if 'WELL' in las.well:
                metadata['well'] = las.well['WELL'].value

            # Field name
            if 'FLD' in las.well:
                metadata['field'] = las.well['FLD'].value
            elif 'FIELD' in las.well:
                metadata['field'] = las.well['FIELD'].value

            # Company
            if 'COMP' in las.well:
                metadata['company'] = las.well['COMP'].value
            elif 'COMPANY' in las.well:
                metadata['company'] = las.well['COMPANY'].value

            # Depth range
            if hasattr(las, 'start') and hasattr(las, 'stop'):
                metadata['depth_range'] = f"{las.start:.2f} - {las.stop:.2f} {las.well['STRT'].unit}"

            # Get curve names
            metadata['curves'] = [curve.mnemonic for curve in las.curves]

            return metadata
        except Exception as e:
            print(f"Error extracting LAS metadata: {str(e)}")
            return {}