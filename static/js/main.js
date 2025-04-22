document.addEventListener('DOMContentLoaded', function() {
    // DOM elements cache
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
        toolMessages: document.getElementById('tool-messages'),
        uploadBtn: document.getElementById('upload-btn'),
        uploadArea: document.getElementById('upload-area'),
        toggleMetadata: document.getElementById('toggle-metadata'),
        metadataContent: document.getElementById('metadata-content'),
        metadataDetails: document.getElementById('metadata-details'),
        curveList: document.getElementById('curve-list'),
        toggleDebug: document.getElementById('toggle-debug'),
        debugPanel: document.getElementById('debug-panel'),
        streamingToggle: document.getElementById('streaming-toggle'),
        suggestionChips: document.querySelectorAll('.suggestion-chip'),
        fileItems: document.querySelectorAll('.document-panel__file-item')
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

        messageDiv.style.height = 'auto';
        messageDiv.style.minHeight = 'fit-content';

        scrollToBottom();
        return messageDiv;
    }

    function showLoadingMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        messageDiv.style.maxHeight = 'none';
        messageDiv.style.overflowY = 'visible';
        messageDiv.style.height = 'auto';

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

    function updateDebugInfo(data) {
        // token usage
        if (data.token_usage) {
            const usage = data.token_usage;
            const cost = data.token_cost;
            elements.tokenUsage.innerHTML = `
                <h4>Token Usage</h4>
                <p>Input: ${usage.input} / Output: ${usage.output} / Total: ${usage.total}</p>
                <p>Cost: $${cost.input.toFixed(2)} / $${cost.output.toFixed(2)} / $${cost.total.toFixed(2)}</p>
            `;
        }

        // thinking process
        if (data.thinking_process) {
            elements.thinkingProcess.innerHTML = `
                <h4>Thinking Process</h4>
                <pre>${data.thinking_process}</pre>
            `;
        }

        // tool messages
        if (data.tool_messages) {
            // Format each tool message individually for better readability
            const formattedMessages = data.tool_messages.map(msg => {
                let content = msg.content;
                try {
                    if (typeof content === 'string' && (content.startsWith('{') || content.startsWith('['))) {
                        // Parse and re-stringify with indentation
                        const parsedContent = JSON.parse(content);
                        content = JSON.stringify(parsedContent, null, 2);
                    }
                } catch (e) {
                    // If parsing fails, keep the original content
                }

                return {
                    name: msg.name,
                    content: content,
                    id: msg.id
                };
            });

            // Create a more structured display for tool messages
            let toolMessagesHTML = `<h4>Tool Messages</h4>`;

            // Create a separate section for each tool message
            formattedMessages.forEach((msg, index) => {
                toolMessagesHTML += `
                <div class="tool-message">
                    <div class="tool-name">Tool: ${msg.name}</div>
                    <pre class="tool-content">${msg.content}</pre>
                </div>
                ${index < formattedMessages.length - 1 ? '<hr>' : ''}
                `;
            });

            elements.toolMessages.innerHTML = toolMessagesHTML;
        }
    }

    function resetDebugInfo() {
        elements.tokenUsage.innerHTML = '<h4>Token Usage</h4><p>No data yet</p>';
        elements.thinkingProcess.innerHTML = '<h4>Thinking Process</h4><pre>No data yet</pre>';
        elements.toolMessages.innerHTML = '<h4>Tool Messages</h4><div class="tool-message"><div class="tool-name">Status</div><pre class="tool-content">No tool messages yet</pre></div>';
    }

    function displayResponse(responseDiv, data) {
        responseDiv.innerHTML = '';

        // Create a container for all visualizations
        const visualizationsContainer = document.createElement('div');
        visualizationsContainer.className = 'visualizations-wrapper';

        let hasVisualizations = false;

        // Display interactive Plotly visualizations
        if (data.plotly_visualizations && data.plotly_visualizations.length > 0) {
            data.plotly_visualizations.forEach((viz, index) => {
                hasVisualizations = true;

                // Create a container for this visualization
                const vizContainer = document.createElement('div');
                vizContainer.className = 'visualization-container';
                vizContainer.style.width = '100%';
                vizContainer.style.maxWidth = '100%';
                vizContainer.style.height = 'auto';

                // Add a title for the visualization
                const vizTitle = document.createElement('div');
                vizTitle.className = 'visualization-title';
                vizTitle.textContent = viz.visualization_type ?
                    `${viz.visualization_type.charAt(0).toUpperCase() + viz.visualization_type.slice(1)} Visualization` :
                    `Visualization ${index + 1}`;
                vizContainer.appendChild(vizTitle);

                // Create a container for the Plotly visualization
                const plotContainer = document.createElement('div');
                const cleanId = viz.plot_id.replace(/[^a-zA-Z0-9_]/g, '_');
                plotContainer.id = cleanId;
                plotContainer.className = 'plotly-visualization';
                vizContainer.appendChild(plotContainer);
                visualizationsContainer.appendChild(vizContainer);

                // Render the Plotly visualization
                setTimeout(() => {
                    try {
                        const plotElement = document.getElementById(cleanId);
                        if (plotElement && typeof Plotly !== 'undefined' &&
                            Array.isArray(viz.plot_data) && viz.plot_data.length > 0) {

                            // Set layout
                            const layout = viz.plot_layout || {};

                            // Calculate container dimensions
                            const containerWidth = plotElement.parentElement.clientWidth - 40; // Subtract padding

                            // Set dimensions
                            layout.width = layout.width || Math.min(containerWidth, 0.95 * window.innerWidth);
                            layout.height = layout.height || 450; // Default height
                            layout.autosize = true; // Allow autosize for height

                            // Ensure margins aren't too large but provide enough space
                            layout.margin = layout.margin || {};
                            layout.margin.l = layout.margin.l || 60;
                            layout.margin.r = layout.margin.r || 60;
                            layout.margin.t = layout.margin.t || 60;
                            layout.margin.b = layout.margin.b || 60;
                            layout.margin.pad = layout.margin.pad || 10;

                            // Center the title
                            if (layout.title) {
                                layout.title = {
                                    text: layout.title.text || layout.title,
                                    x: 0.5,  // Center the title
                                    xanchor: 'center'
                                };
                            }

                            // Render the plot
                            Plotly.newPlot(cleanId, viz.plot_data, layout, {
                                responsive: true,
                                scrollZoom: true,
                                displayModeBar: true,
                                displaylogo: false,
                                modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
                                showAxisDragHandles: true,
                                fillFrame: true
                            })
                            .then(function() {
                                // After plot is created, adjust the container height to fit the plot
                                const plotHeight = plotElement.getBoundingClientRect().height;
                                if (plotHeight > 0) {
                                    // Add some padding to the container height
                                    plotElement.parentElement.style.height = (plotHeight + 40) + 'px';
                                }
                            });

                            // Add window resize handler to ensure the plot stays within its container
                            const resizeHandler = function() {
                                try {
                                    const newWidth = plotElement.parentElement.clientWidth - 40;
                                    Plotly.relayout(cleanId, {
                                        width: newWidth
                                    })
                                    .then(function() {
                                        // After resize, adjust the container height again
                                        const plotHeight = plotElement.getBoundingClientRect().height;
                                        if (plotHeight > 0) {
                                            plotElement.parentElement.style.height = (plotHeight + 40) + 'px';
                                        }
                                    });
                                } catch (e) {
                                    // Silent error handling
                                }
                            };

                            window.addEventListener('resize', resizeHandler);
                        }
                    } catch (error) {
                        // Silent error handling
                    }
                }, 100);
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

        // Ensure the response div expands to fit its content
        responseDiv.style.height = 'auto';
        responseDiv.style.minHeight = 'fit-content';
        responseDiv.style.overflowY = 'visible';

        scrollToBottom();
    }

    function handleFileItemClick() {
        document.querySelectorAll('.document-panel__file-item').forEach(item => {
            item.classList.remove('document-panel__file-item--active');
        });

        this.classList.add('document-panel__file-item--active');

        // Get file info from data attribute
        if (this.dataset.fileInfo) {
            const fileInfo = JSON.parse(this.dataset.fileInfo);
            updateMetadataPanel(fileInfo);
        }
    }

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const uploadMessageDiv = showLoadingMessage('Uploading file...');

        // Check if file with same name already exists
        const existingFile = Array.from(elements.filesList.children).find(item =>
            item.dataset.filename === file.name
        );

        if (existingFile) {
            // If file already exists, show a message and highlight it
            uploadMessageDiv.innerHTML = `File <strong>${file.name}</strong> is already uploaded.`;

            // Highlight the existing file
            document.querySelectorAll('.document-panel__file-item').forEach(item => {
                item.classList.remove('document-panel__file-item--active');
            });
            existingFile.classList.add('document-panel__file-item--active');

            // Update metadata panel with the existing file's info
            if (existingFile.dataset.fileInfo) {
                const fileInfo = JSON.parse(existingFile.dataset.fileInfo);
                updateMetadataPanel(fileInfo);
            }

            return;
        }

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                uploadMessageDiv.innerHTML = `File <strong>${data.file_name}</strong> uploaded successfully.`;

                const fileItem = document.createElement('li');
                fileItem.className = 'document-panel__file-item';
                fileItem.dataset.filename = data.file_name;

                // Store file info in data attribute to avoid redundant API calls
                if (data.file_info) {
                    fileItem.dataset.fileInfo = JSON.stringify(data.file_info);
                }

                // Make file item clickable using the shared handler function
                fileItem.addEventListener('click', handleFileItemClick);

                const fileIcon = document.createElement('i');
                fileIcon.className = 'fas fa-file-alt document-panel__file-icon';

                const fileName = document.createElement('span');
                fileName.className = 'document-panel__file-name';
                fileName.textContent = data.file_name;

                fileItem.appendChild(fileIcon);
                fileItem.appendChild(fileName);
                elements.filesList.appendChild(fileItem);
                elements.fileInput.value = '';

                // Update metadata panel with file info
                if (data.file_info && elements.metadataDetails) {
                    updateMetadataPanel(data.file_info);

                    // Remove active class from all other file items
                    document.querySelectorAll('.document-panel__file-item').forEach(item => {
                        if (item !== fileItem) {
                            item.classList.remove('document-panel__file-item--active');
                        }
                    });

                    // Mark this file as active
                    fileItem.classList.add('document-panel__file-item--active');
                }
            } else {
                uploadMessageDiv.innerHTML = `Error: ${data.message || 'Unknown error'}`;
            }
        })
        .catch(() => {
            uploadMessageDiv.innerHTML = 'Upload failed. Please try again.';
        });
    }

    function updateMetadataPanel(fileInfo) {
        // Hide empty state
        const emptyState = document.getElementById('metadata-empty-state');
        if (emptyState) {
            emptyState.style.display = 'none';
        }

        // Update metadata details
        const metadataDetails = document.getElementById('metadata-details');
        if (metadataDetails) {
            let detailsHTML = '<h4 class="metadata-panel__subtitle">File Details</h4>';

            if (fileInfo.well) {
                detailsHTML += `<p><strong>Well:</strong> ${fileInfo.well}</p>`;
            }

            if (fileInfo.field) {
                detailsHTML += `<p><strong>Field:</strong> ${fileInfo.field}</p>`;
            }

            if (fileInfo.company) {
                detailsHTML += `<p><strong>Company:</strong> ${fileInfo.company}</p>`;
            }

            if (fileInfo.depth_range) {
                detailsHTML += `<p><strong>Depth Range:</strong> ${fileInfo.depth_range}</p>`;
            }

            metadataDetails.innerHTML = detailsHTML;
            metadataDetails.style.display = 'block';
        }

        // Update curve list
        const curvesSection = document.getElementById('metadata-curves');
        const curveList = document.getElementById('curve-list');

        if (curvesSection && curveList && fileInfo.curves && fileInfo.curves.length > 0) {
            // Show the curves section
            curvesSection.style.display = 'block';
            curveList.innerHTML = '';

            fileInfo.curves.forEach(curve => {
                const curveItem = document.createElement('li');
                curveItem.className = 'metadata-panel__curve-item';
                curveItem.textContent = curve;
                curveItem.addEventListener('click', () => {
                    elements.queryInput.value = `Tell me about the ${curve} curve`;
                    sendQuery();
                });

                curveList.appendChild(curveItem);
            });
        } else if (curvesSection) {
            // Hide the curves section if no curves are available
            curvesSection.style.display = 'none';
        }

        // Show metadata panel
        const metadataContent = document.getElementById('metadata-content');
        if (metadataContent) {
            metadataContent.style.display = 'block';
        }
    }

    // Add streaming toggle state
    let streamingEnabled = false;

    // Add event listener for the streaming toggle
    if (elements.streamingToggle) {
        elements.streamingToggle.addEventListener('change', function() {
            streamingEnabled = this.checked;
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

        if (streamingEnabled) {
            const q = encodeURIComponent(query);
            let responseText = '';

            // Stream the text response using custom event stream format
            let hasReceivedVisualizations = false;

            function connectToEventStream() {
                fetch(`/query_stream?query=${q}`, {
                        headers: {
                            'Accept': 'text/event-stream'
                        }
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP error! Status: ${response.status}`);
                        }

                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        let buffer = "";

                        function processStream({ done, value }) {
                        if (done) {
                            // If we still have data in the buffer, try to process it
                            const trimmedBuffer = buffer.trim();
                            if (trimmedBuffer) {
                                try {
                                    handleEvent(JSON.parse(trimmedBuffer));
                                } catch (e) {
                                }
                            }

                            // If no visualizations were received, just make sure the text is formatted
                            if (!hasReceivedVisualizations) {
                                loadingMessageDiv.innerHTML = formatResponseText(responseText);
                            }
                            scrollToBottom();
                            return;
                        }

                        // Decode the chunk and add it to our buffer
                        const chunk = decoder.decode(value, { stream: true });
                        buffer += chunk;

                        // Process complete JSON objects in the buffer
                        let newlineIndex;
                        while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                            const eventJson = buffer.slice(0, newlineIndex);
                            buffer = buffer.slice(newlineIndex + 1);

                            const trimmedJson = eventJson.trim();
                            if (trimmedJson) {
                                try {
                                    handleEvent(JSON.parse(trimmedJson));
                                } catch (e) {
                                }
                            }
                        }

                        // Continue reading the stream
                        return reader.read().then(processStream);
                    }

                    // Handler for different event types
                    function handleEvent(event) {
                        if (!event?.event) return;

                        switch (event.event) {
                            case 'text_chunk':
                                if (event.data?.text) {
                                    const textChunk = event.data.text;
                                    responseText += textChunk;

                                    // Only start displaying content once we have a reasonable amount of text
                                    if (responseText.length > 10) {
                                        // Use a more efficient approach for updating the content
                                        if (textChunk.includes('\n')) {
                                            // Full reformat with markdown for new paragraphs
                                            loadingMessageDiv.innerHTML = formatResponseText(responseText);
                                        } else {
                                            // Simple append for most chunks to improve performance
                                            const lastChild = loadingMessageDiv.lastChild;
                                            if (lastChild && lastChild.nodeType === Node.ELEMENT_NODE) {
                                                lastChild.innerHTML += textChunk;
                                            } else {
                                                loadingMessageDiv.innerHTML = formatResponseText(responseText);
                                            }
                                        }
                                        scrollToBottom();
                                    }
                                }
                                break;
                            case 'visualizations':
                                if (event.data?.plotly_visualizations?.length > 0) {
                                    hasReceivedVisualizations = true;
                                    // Create a data object in the format expected by displayResponse
                                    const responseData = {
                                        response: responseText,
                                        plotly_visualizations: event.data.plotly_visualizations
                                    };

                                    // Display the response with visualizations
                                    displayResponse(loadingMessageDiv, responseData);
                                    scrollToBottom();
                                }
                                break;
                            case 'tool_messages':
                                if (event.data?.tool_messages) {
                                    // Store tool messages for later use
                                    updateDebugInfo({ tool_messages: event.data.tool_messages });
                                }
                                break;
                            case 'complete':
                                if (event.data) {
                                    // Update debug panel with the received information
                                    updateDebugInfo(event.data);

                                    // If we haven't received visualizations yet, just format the text response
                                    if (!hasReceivedVisualizations) {
                                        loadingMessageDiv.innerHTML = formatResponseText(responseText);
                                    }
                                    scrollToBottom();
                                }
                                break;
                            case 'error':
                                if (event.data?.message) {
                                    // Clear any existing content and show the error message
                                    loadingMessageDiv.innerHTML = '';
                                    const errorDiv = document.createElement('div');
                                    errorDiv.className = 'error-message';
                                    errorDiv.textContent = `Error: ${event.data.message}`;
                                    loadingMessageDiv.appendChild(errorDiv);
                                }
                                break;
                            default:
                        }
                    }
                    return reader.read().then(processStream);
                })
                .catch(error => {
                    loadingMessageDiv.innerHTML = `<div class="error-message">Error during streaming: ${error.message}</div>`;
                });
            }

            // Start the connection
            connectToEventStream();
        } else {
            // Use regular endpoint
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
    }

    // Function to reset the metadata panel
    function resetMetadataPanel() {
        // Show empty state message
        const emptyState = document.getElementById('metadata-empty-state');
        if (emptyState) {
            emptyState.style.display = 'block';
        }

        // Clear metadata details
        const metadataDetails = document.getElementById('metadata-details');
        if (metadataDetails) {
            metadataDetails.innerHTML = '';
            metadataDetails.style.display = 'none';
        }

        // Clear and hide curves section
        const curvesSection = document.getElementById('metadata-curves');
        const curveList = document.getElementById('curve-list');
        if (curvesSection) {
            curvesSection.style.display = 'none';
        }
        if (curveList) {
            curveList.innerHTML = '';
        }
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

                // Reset metadata panel (Well Information)
                resetMetadataPanel();
            }
        })
        .catch(() => {
            alert('An error occurred while clearing the chat');
        });
    }

    // Handle file selection and automatically submit the form
    elements.fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            // Automatically submit the form when a file is selected
            elements.uploadForm.dispatchEvent(new Event('submit'));
        }
    });

    // Handle form submission
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

    // Toggle metadata panel
    if (elements.toggleMetadata) {
        elements.toggleMetadata.addEventListener('click', function() {
            const isExpanded = elements.metadataContent.style.display !== 'none';
            elements.metadataContent.style.display = isExpanded ? 'none' : 'block';
            elements.toggleMetadata.innerHTML = isExpanded ?
                '<i class="fas fa-chevron-down"></i>' :
                '<i class="fas fa-chevron-up"></i>';
        });
    }

    // Handle upload button click
    if (elements.uploadBtn) {
        elements.uploadBtn.addEventListener('click', function() {
            elements.fileInput.click();
        });
    }

    // Handle drag and drop for file upload
    if (elements.uploadArea) {
        // Create a reusable function for handling file drops
        function handleFileDrop(e) {
            e.preventDefault();
            e.stopPropagation();
            elements.uploadArea.classList.remove('document-panel__upload-area--active');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                // Set the file input's files property
                elements.fileInput.files = files;
                // Upload the first file directly
                uploadFile(files[0]);

                // If multiple files were dropped, upload them one by one
                if (files.length > 1) {
                    for (let i = 1; i < files.length; i++) {
                        setTimeout(() => {
                            uploadFile(files[i]);
                        }, i * 1000); // Stagger uploads by 1 second each
                    }
                }
            }
        }

        // Prevent default behavior for dragover to allow drop
        elements.uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.add('document-panel__upload-area--active');
        });

        // Handle drag leave
        elements.uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('document-panel__upload-area--active');
        });

        // Handle drop event
        elements.uploadArea.addEventListener('drop', handleFileDrop);

        // Also add the same handlers to the upload form and label for better coverage
        const uploadLabel = document.querySelector('.document-panel__upload-label');
        if (uploadLabel) {
            uploadLabel.addEventListener('dragover', function(e) {
                e.preventDefault();
                e.stopPropagation();
                elements.uploadArea.classList.add('document-panel__upload-area--active');
            });

            uploadLabel.addEventListener('dragleave', function(e) {
                e.preventDefault();
                e.stopPropagation();
                elements.uploadArea.classList.remove('document-panel__upload-area--active');
            });

            // Use the same drop handler for the label
            uploadLabel.addEventListener('drop', handleFileDrop);
        }
    }

    // Handle suggestion chips
    if (elements.suggestionChips) {
        elements.suggestionChips.forEach(chip => {
            chip.addEventListener('click', function() {
                elements.queryInput.value = this.textContent;
                sendQuery();
            });
        });
    }

    // Auto-resize textarea
    function autoResizeTextarea() {
        elements.queryInput.style.height = 'auto';
        elements.queryInput.style.height = (elements.queryInput.scrollHeight) + 'px';
    }

    // Helper function to format response text with markdown or fallback to HTML
    function formatResponseText(text) {
        // Check if marked is available, otherwise fallback to simple line breaks
        if (typeof marked !== 'undefined') {
            try {
                return marked.parse(text);
            } catch (e) {
                console.warn('Error parsing markdown:', e);
                return text.replace(/\n/g, '<br>');
            }
        } else {
            // Fallback if marked is not available
            return text.replace(/\n/g, '<br>');
        }
    }

    elements.queryInput.addEventListener('input', autoResizeTextarea);

    // Collapsible sections
    const collapsibles = document.getElementsByClassName('collapsible');
    for (const collapsible of collapsibles) {
        collapsible.addEventListener('click', function() {
            this.classList.toggle('active');
            const content = this.nextElementSibling;
            content.style.display = content.style.display === 'block' ? 'none' : 'block';
        });
    }

    scrollToBottom();

    // Hide debug panel by default
    if (elements.debugPanel) {
        elements.debugPanel.style.display = 'none';
    }

    // Add toggle debug panel functionality
    if (elements.toggleDebug && elements.debugPanel) {
        elements.toggleDebug.addEventListener('click', function() {
            // Toggle debug panel visibility
            const isVisible = elements.debugPanel.style.display === 'block';
            elements.debugPanel.style.display = isVisible ? 'none' : 'block';

            // Toggle active class on the settings button
            if (isVisible) {
                elements.toggleDebug.classList.remove('settings-button--active');
            } else {
                elements.toggleDebug.classList.add('settings-button--active');

                // Auto-expand the first collapsible section when showing the panel
                const collapsible = elements.debugPanel.querySelector('.collapsible');
                const content = collapsible?.nextElementSibling;
                if (collapsible && content && content.style.display !== 'block') {
                    collapsible.classList.add('active');
                    content.style.display = 'block';
                }
            }
        });
    }

    // Initialize metadata panel
    if (elements.metadataContent && elements.filesList && elements.filesList.children.length === 0) {
        const curvesSection = document.getElementById('metadata-curves');
        if (curvesSection) {
            curvesSection.style.display = 'none';
        }
    }

    // Add click handlers to existing file items
    if (elements.fileItems && elements.fileItems.length > 0) {
        // Add click handlers to all file items using the shared handler function
        elements.fileItems.forEach(fileItem => {
            const fileName = fileItem.querySelector('.document-panel__file-name').textContent;
            fileItem.dataset.filename = fileName;
            fileItem.addEventListener('click', handleFileItemClick);
        });

        // Select the first file by default
        if (elements.fileItems.length > 0) {
            elements.fileItems[0].click();
        }
    }
});