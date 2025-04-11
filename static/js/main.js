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

        // Create a container for all visualizations
        const visualizationsContainer = document.createElement('div');
        visualizationsContainer.className = 'visualizations-wrapper';

        let hasVisualizations = false;

        // No legacy static visualizations

        // Display interactive Plotly visualizations
        if (data.plotly_visualizations && data.plotly_visualizations.length > 0) {
            data.plotly_visualizations.forEach((viz, index) => {
                hasVisualizations = true;

                // Create a container for this visualization
                const vizContainer = document.createElement('div');
                vizContainer.className = 'visualization-container';

                // Add a title for the visualization
                const vizTitle = document.createElement('div');
                vizTitle.className = 'visualization-title';
                vizTitle.textContent = viz.visualization_type ?
                    `${viz.visualization_type.charAt(0).toUpperCase() + viz.visualization_type.slice(1)} Visualization` :
                    `Visualization ${index + 1}`;
                vizContainer.appendChild(vizTitle);

                // Create a container for the Plotly visualization
                const plotContainer = document.createElement('div');

                // Clean the ID to ensure it's valid for DOM
                // Remove any special characters that might cause issues
                const cleanId = viz.plot_id.replace(/[^a-zA-Z0-9_]/g, '_');

                // Ensure the ID is exactly as expected by Plotly
                plotContainer.id = cleanId;
                plotContainer.className = 'plotly-visualization';

                // Store both the original and cleaned IDs for debugging
                plotContainer.setAttribute('data-original-plot-id', viz.plot_id);
                plotContainer.setAttribute('data-clean-plot-id', cleanId);

                // Store the cleaned ID back in the viz object for later use
                viz.clean_plot_id = cleanId;

                vizContainer.appendChild(plotContainer);



                // Add the container to the visualizations wrapper
                visualizationsContainer.appendChild(vizContainer);



                // Render the Plotly visualization after a short delay to ensure DOM is ready
                setTimeout(() => {
                    try {
                        // Use the cleaned ID to find the element
                        const plotId = viz.clean_plot_id || viz.plot_id.replace(/[^a-zA-Z0-9_]/g, '_');

                        // Get the plot element
                        const plotElement = document.getElementById(plotId);
                        if (!plotElement || typeof Plotly === 'undefined') {
                            return;
                        }

                        // Ensure the container has appropriate dimensions for plotting
                        plotElement.style.minHeight = '350px';
                        plotElement.style.height = '400px';
                        plotElement.style.maxWidth = '100%';
                        plotElement.style.border = '1px solid #ddd';
                        plotElement.style.backgroundColor = '#f9f9f9';

                        // Add a border to make the plot container visible
                        plotElement.style.border = '1px solid #ddd';

                        // Skip if plot data is invalid
                        if (!Array.isArray(viz.plot_data) || viz.plot_data.length === 0) {
                            return;
                        }

                        // Ensure layout has proper size settings
                        const layout = viz.plot_layout || {};
                        layout.height = layout.height || 380;
                        layout.width = layout.width || Math.min(800, 0.75 * window.innerWidth); // Cap at 800px or 75% of window width
                        layout.autosize = true;
                        layout.margin = layout.margin || {l: 50, r: 50, t: 50, b: 50}; // Reduce margins

                        // Render the plot with updated layout
                        Plotly.newPlot(plotId, viz.plot_data, layout, {responsive: true, useResizeHandler: true})
                            .catch(() => {});
                            // Silent error handling - visualization will simply not appear if there's an error
                    } catch (error) {
                        // Silent error handling
                    }
                }, 200); // Increased timeout to ensure DOM is ready
            });
        }

        // Add visualizations to the response if there are any
        if (hasVisualizations) {
            responseDiv.appendChild(visualizationsContainer);
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
    for (const collapsible of collapsibles) {
        collapsible.addEventListener('click', function() {
            this.classList.toggle('active');
            const content = this.nextElementSibling;
            content.style.display = content.style.display === 'block' ? 'none' : 'block';
        });
    }

    scrollToBottom();
});