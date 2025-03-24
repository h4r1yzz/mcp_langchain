class StateManager:
    """Manages application state separately from the UI."""
    
    def __init__(self, session_state):
        """Initialize with Streamlit's session_state."""
        self.session_state = session_state
        self._initialize_state()
        
    def _initialize_state(self):
        """Initialize state variables if they don't exist."""
        if "messages" not in self.session_state:
            self.session_state.messages = []
            
        if "file_path" not in self.session_state:
            self.session_state.file_path = None
            
        if "uploaded_files" not in self.session_state:
            self.session_state.uploaded_files = {}
            
        if "thinking_process" not in self.session_state:
            self.session_state.thinking_process = ""
            
        if "token_usage" not in self.session_state:
            self.session_state.token_usage = {"input": 0, "output": 0, "total": 0}
    
    # Message methods
    def get_messages(self):
        return self.session_state.messages
        
    def add_message(self, role, content):
        self.session_state.messages.append({"role": role, "content": content})
        
    def clear_messages(self):
        self.session_state.messages = []
    
    # File methods
    def get_file_path(self):
        return self.session_state.file_path
        
    def set_file_path(self, path):
        self.session_state.file_path = path
        
    def get_uploaded_files(self):
        return self.session_state.uploaded_files
        
    def add_uploaded_file(self, name, path):
        self.session_state.uploaded_files[name] = {
            "name": name,
            "path": path
        }
    
    # Thinking process methods
    def get_thinking_process(self):
        return self.session_state.thinking_process
        
    def set_thinking_process(self, process):
        self.session_state.thinking_process = process
    
    # Token usage methods
    def get_token_usage(self):
        return self.session_state.token_usage
        
    def set_token_usage(self, usage):
        self.session_state.token_usage = usage
        
    def clear_all(self):
        """Clear all state."""
        self.clear_messages()
        self.session_state.thinking_process = ""
        self.session_state.token_usage = {"input": 0, "output": 0, "total": 0}
