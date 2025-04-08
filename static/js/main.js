document.addEventListener('DOMContentLoaded', function() {
    const elements = {
        chatContainer: document.getElementById('chat-container'),
        queryInput: document.getElementById('query-input'),
        sendButton: document.getElementById('send-button'),
        uploadForm: document.getElementById('upload-form'),
        fileInput: document.getElementById('file-input'),
        filesList: document.getElementById('files-list'),
        clearChatButton: document.getElementById('clear-chat'),
        tokenUsage: document.getElementById('token-usage'),
        thinkingProcess: document.getElementById('thinking-process'),
        toolMessages: document.getElementById('tool-messages')
    };

    let loadingMessageDiv = null;

    function scrollToBottom() {
        elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    }

    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        if (role === 'assistant') {
            content = content.replace(/\n/g, '<br>');
        }

        messageDiv.innerHTML = content;
        elements.chatContainer.appendChild(messageDiv);
        scrollToBottom();
        return messageDiv;
    }

    function showLoadingMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        const loadingDiv = document.createElement('div');
        loadingDiv.textContent = text;

        const spinner = document.createElement('div');
        spinner.className = 'inline-spinner';

        loadingDiv.appendChild(spinner);
        messageDiv.appendChild(loadingDiv);

        elements.chatContainer.appendChild(messageDiv);
        scrollToBottom();

        return messageDiv;
    }

    function showErrorMessage(errorText) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message';

        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = errorText || 'An error occurred';

        messageDiv.appendChild(errorDiv);
        elements.chatContainer.appendChild(messageDiv);
        scrollToBottom();
    }

    function updateDebugInfo(data) {
        // Update token usage
        if (data.token_usage) {
            const usage = data.token_usage;
            const cost = data.token_cost;
            elements.tokenUsage.innerHTML = `
                <h4>Token Usage</h4>
                <p>Input: ${usage.input} / Output: ${usage.output} / Total: ${usage.total}</p>
                <p>Cost: $${cost.input.toFixed(2)} / $${cost.output.toFixed(2)} / $${cost.total.toFixed(2)}</p>
            `;
        }

        // Update thinking process
        if (data.thinking_process) {
            elements.thinkingProcess.innerHTML = `
                <h4>Thinking Process</h4>
                <pre>${data.thinking_process}</pre>
            `;
        }

        // Update tool messages
        if (data.tool_messages) {
            elements.toolMessages.innerHTML = `
                <h4>Tool Messages</h4>
                <pre>${JSON.stringify(data.tool_messages, null, 2)}</pre>
            `;
        }
    }

    function resetDebugInfo() {
        elements.tokenUsage.innerHTML = '<h4>Token Usage</h4><p>No data yet</p>';
        elements.thinkingProcess.innerHTML = '<h4>Thinking Process</h4><pre>No data yet</pre>';
        elements.toolMessages.innerHTML = '<h4>Tool Messages</h4><pre>No data yet</pre>';
    }

    function displayResponse(responseDiv, data) {
        responseDiv.innerHTML = '';

        if (data.visualizations && data.visualizations.length > 0) {
            data.visualizations.forEach(viz => {
                const imgElement = document.createElement('img');
                imgElement.src = viz.url;
                imgElement.className = 'visualization';
                imgElement.alt = 'Visualization';
                responseDiv.appendChild(imgElement);
            });
        }

        // FORMATTED RESPONSE
        const formattedResponse = data.response.replace(/\n/g, '<br>');
        const textDiv = document.createElement('div');
        textDiv.className = 'response-text';
        textDiv.innerHTML = formattedResponse;
        responseDiv.appendChild(textDiv);

        scrollToBottom();
    }

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const uploadMessageDiv = showLoadingMessage('Uploading file...');

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                uploadMessageDiv.innerHTML = `File <strong>${data.file_name}</strong> uploaded successfully.`;

                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';

                const fileIcon = document.createElement('span');
                fileIcon.className = 'file-icon';
                fileIcon.textContent = '📄';

                const fileName = document.createElement('span');
                fileName.textContent = data.file_name;

                fileItem.appendChild(fileIcon);
                fileItem.appendChild(fileName);
                elements.filesList.appendChild(fileItem);
                elements.fileInput.value = '';
            } else {
                uploadMessageDiv.innerHTML = '';
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = `Error: ${data.message}`;
                uploadMessageDiv.appendChild(errorDiv);
            }
        })
        .catch(() => {
            uploadMessageDiv.innerHTML = '';
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = 'An error occurred while processing your request';
            uploadMessageDiv.appendChild(errorDiv);
        });
    }

    function sendQuery() {
        const query = elements.queryInput.value.trim();
        if (!query) return;

        // Add user message
        addMessage('user', query);
        elements.queryInput.value = '';

        // Show loading message
        loadingMessageDiv = showLoadingMessage('Generating Response...');

        // Send query to server
        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                displayResponse(loadingMessageDiv, data);
                updateDebugInfo(data);
            } else {
                loadingMessageDiv.remove();
                showErrorMessage(data.message || 'An error occurred');
            }
        })
        .catch(() => {
            loadingMessageDiv.remove();
            showErrorMessage('An error occurred while processing your request');
        });
    }

    function clearChat() {
        fetch('/clear', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Clear chat messages
                elements.chatContainer.innerHTML = '';

                // Clear uploaded files list
                elements.filesList.innerHTML = '';

                // Reset debug info
                resetDebugInfo();
            }
        })
        .catch(() => {
            alert('An error occurred while clearing the chat');
        });
    }

    elements.uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const file = elements.fileInput.files[0];
        if (!file) {
            alert('Please select a file');
            return;
        }
        uploadFile(file);
    });

    elements.sendButton.addEventListener('click', sendQuery);

    elements.queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendQuery();
        }
    });

    elements.clearChatButton.addEventListener('click', clearChat);

    const collapsibles = document.getElementsByClassName('collapsible');
    for (let i = 0; i < collapsibles.length; i++) {
        collapsibles[i].addEventListener('click', function() {
            this.classList.toggle('active');
            const content = this.nextElementSibling;
            content.style.display = content.style.display === 'block' ? 'none' : 'block';
        });
    }

    scrollToBottom();
});