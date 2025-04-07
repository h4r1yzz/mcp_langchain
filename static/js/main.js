document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const filesList = document.getElementById('files-list');
    const clearChatButton = document.getElementById('clear-chat');
    const spinner = document.getElementById('spinner');
    const tokenUsage = document.getElementById('token-usage');
    const thinkingProcess = document.getElementById('thinking-process');
    const toolMessages = document.getElementById('tool-messages');

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.innerHTML = content;
        chatContainer.appendChild(messageDiv);
        scrollToBottom();
    }

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        spinner.style.display = 'block';

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            spinner.style.display = 'none';

            if (data.status === 'success') {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <span class="file-icon">📄</span>
                    <span>${data.file_name}</span>
                `;
                filesList.appendChild(fileItem);
                fileInput.value = '';
                alert(`File uploaded: ${data.file_name}`);
            } else {
                alert(`Error: ${data.message}`);
            }
        })
        .catch(() => {
            spinner.style.display = 'none';
            alert('An error occurred while processing your request');
        });
    });

    sendButton.addEventListener('click', sendQuery);

    queryInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendQuery();
        }
    });

    function sendQuery() {
        const query = queryInput.value.trim();
        if (!query) return;

        addMessage('user', query);
        queryInput.value = '';
        spinner.style.display = 'block';

        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        })
        .then(response => response.json())
        .then(data => {
            spinner.style.display = 'none';

            if (data.status === 'success') {
                addMessage('assistant', data.response);

                if (data.token_usage) {
                    const usage = data.token_usage;
                    const cost = data.token_cost;
                    tokenUsage.innerHTML = `
                        <h4>Token Usage</h4>
                        <p>Input: ${usage.input} / Output: ${usage.output} / Total: ${usage.total}</p>
                        <p>Cost: $${cost.input.toFixed(2)} / $${cost.output.toFixed(2)} / $${cost.total.toFixed(2)}</p>
                    `;
                }

                if (data.thinking_process) {
                    thinkingProcess.innerHTML = `
                        <h4>Thinking Process</h4>
                        <pre>${data.thinking_process}</pre>
                    `;
                }

                if (data.tool_messages) {
                    toolMessages.innerHTML = `
                        <h4>Tool Messages</h4>
                        <pre>${JSON.stringify(data.tool_messages, null, 2)}</pre>
                    `;
                }

                if (data.visualizations && data.visualizations.length > 0) {
                    data.visualizations.forEach(viz => {
                        const imgElement = document.createElement('img');
                        imgElement.src = viz.url;
                        imgElement.className = 'visualization';
                        imgElement.alt = 'Visualization';
                        chatContainer.appendChild(imgElement);
                    });
                    scrollToBottom();
                }
            } else {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message';

                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.textContent = data.message || 'An error occurred';

                messageDiv.appendChild(errorDiv);
                chatContainer.appendChild(messageDiv);
                scrollToBottom();
            }
        })
        .catch(() => {
            spinner.style.display = 'none';
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';

            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = 'An error occurred while processing your request';

            messageDiv.appendChild(errorDiv);
            chatContainer.appendChild(messageDiv);
            scrollToBottom();
        });
    }

    clearChatButton.addEventListener('click', function() {
        fetch('/clear', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                chatContainer.innerHTML = '';

                tokenUsage.innerHTML = '<h4>Token Usage</h4><p>No data yet</p>';
                thinkingProcess.innerHTML = '<h4>Thinking Process</h4><pre>No data yet</pre>';
                toolMessages.innerHTML = '<h4>Tool Messages</h4><pre>No data yet</pre>';
            }
        })
        .catch(() => {
            alert('An error occurred while clearing the chat');
        });
    });

    const coll = document.getElementsByClassName("collapsible");
    for (let i = 0; i < coll.length; i++) {
        coll[i].addEventListener("click", function() {
            this.classList.toggle("active");
            const content = this.nextElementSibling;
            if (content.style.display === "block") {
                content.style.display = "none";
            } else {
                content.style.display = "block";
            }
        });
    }

    scrollToBottom();
});
