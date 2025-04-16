class StateManager:
    def __init__(self, session_state=None):
        self.session_state = session_state if session_state is not None else {}
        self.is_streamlit = not isinstance(self.session_state, dict)
        self._initialize_state()

    def _initialize_state(self):
        """Initialize state variables if they don't exist."""
        self._ensure_key("messages", [])
        self._ensure_key("file_path", None)
        self._ensure_key("uploaded_files", {})
        self._ensure_key("thinking_process", "")
        self._ensure_key("token_usage", {"input": 0, "output": 0, "total": 0})
        self._ensure_key("tool_messages", [])

    def _ensure_key(self, key, default_value):
        """Ensure a key exists in the session state."""
        if self.is_streamlit:
            # Streamlit style
            if key not in self.session_state:
                setattr(self.session_state, key, default_value)
        else:
            # Flask style (or dict for testing)
            if key not in self.session_state:
                self.session_state[key] = default_value

    # Message methods
    def get_messages(self):
        """Get all messages."""
        if self.is_streamlit:
            return self.session_state.messages
        return self.session_state["messages"]

    def add_message(self, role, content):
        """Add a message to the history."""
        if self.is_streamlit:
            self.session_state.messages.append({"role": role, "content": content})
        else:
            self.session_state["messages"].append({"role": role, "content": content})

    def clear_messages(self):
        """Clear all messages."""
        if self.is_streamlit:
            self.session_state.messages = []
        else:
            self.session_state["messages"] = []

    # File methods
    def get_file_path(self):
        """Get the current file path."""
        if self.is_streamlit:
            return self.session_state.file_path
        return self.session_state["file_path"]

    def set_file_path(self, path):
        """Set the current file path."""
        if self.is_streamlit:
            self.session_state.file_path = path
        else:
            self.session_state["file_path"] = path

    def get_uploaded_files(self):
        """Get all uploaded files."""
        if self.is_streamlit:
            return self.session_state.uploaded_files
        return self.session_state["uploaded_files"]

    def add_uploaded_file(self, name, path, metadata=None):
        """Add an uploaded file."""
        file_info = {
            "name": name,
            "path": path,
            "metadata": metadata or {}
        }

        if self.is_streamlit:
            self.session_state.uploaded_files[name] = file_info
        else:
            self.session_state["uploaded_files"][name] = file_info

    # Thinking process methods
    def get_thinking_process(self):
        """Get the thinking process."""
        if self.is_streamlit:
            return self.session_state.thinking_process
        return self.session_state["thinking_process"]

    def set_thinking_process(self, process):
        """Set the thinking process."""
        if self.is_streamlit:
            self.session_state.thinking_process = process
        else:
            self.session_state["thinking_process"] = process

    # Token usage methods
    def get_token_usage(self):
        """Get token usage statistics."""
        if self.is_streamlit:
            return self.session_state.token_usage
        return self.session_state["token_usage"]

    def set_token_usage(self, usage):
        """Set token usage statistics."""
        if self.is_streamlit:
            self.session_state.token_usage = usage
        else:
            self.session_state["token_usage"] = usage

    # Tool message methods
    def get_tool_messages(self):
        """Get tool messages."""
        if self.is_streamlit:
            return self.session_state.tool_messages
        return self.session_state["tool_messages"]

    def set_tool_messages(self, messages):
        """Set tool messages."""
        if self.is_streamlit:
            self.session_state.tool_messages = messages
        else:
            self.session_state["tool_messages"] = messages

    def clear_all(self):
        """Clear all state."""
        self.clear_messages()

        if self.is_streamlit:
            self.session_state.thinking_process = ""
            self.session_state.token_usage = {"input": 0, "output": 0, "total": 0}
            self.session_state.tool_messages = []
            self.session_state.file_path = None
            self.session_state.uploaded_files = {}
        else:
            self.session_state["thinking_process"] = ""
            self.session_state["token_usage"] = {"input": 0, "output": 0, "total": 0}
            self.session_state["tool_messages"] = []
            self.session_state["file_path"] = None
            self.session_state["uploaded_files"] = {}
