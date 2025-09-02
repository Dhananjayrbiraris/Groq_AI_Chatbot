// DOM elements
const chatBox = document.getElementById('chatBox');
const emptyState = document.getElementById('emptyState');
const questionInput = document.getElementById('question');
const searchStatus = document.getElementById('searchStatus');

// Get current timestamp
function getTimestamp() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Toggle web search function
function toggleWebSearch() {
    const webSearchToggle = document.getElementById('webSearchToggle');
    const isEnabled = webSearchToggle ? webSearchToggle.checked : true;
    
    // Update status display
    if (searchStatus) {
        searchStatus.textContent = isEnabled ? 'ON' : 'OFF';
        searchStatus.style.background = isEnabled ? 'rgba(16, 163, 127, 0.1)' : 'rgba(108, 117, 125, 0.1)';
        searchStatus.style.color = isEnabled ? '#0d8c6d' : '#6c757d';
    }
    
    return isEnabled;
}

// Save chat history to localStorage
function saveChatHistory() {
    const messages = chatBox.querySelectorAll('.message');
    const history = Array.from(messages).map(msg => ({
        text: msg.innerText.replace(getTimestamp(), '').trim(),
        sender: msg.classList.contains('user') ? 'user' : 'bot',
        time: msg.querySelector('.timestamp').textContent
    }));
    localStorage.setItem('chatHistory', JSON.stringify(history));
}

// Load chat history from localStorage
function loadChatHistory() {
    const saved = localStorage.getItem('chatHistory');
    if (saved) {
        const history = JSON.parse(saved);
        if (history.length > 0) {
            emptyState.style.display = 'none';
            history.forEach(msg => {
                addMessage(msg.text, msg.sender, msg.time, false);
            });
            scrollToBottom();
        }
    }
}

// Clear chat history from backend
function clearChatHistoryFromBackend() {
    const sessionId = document.getElementById('sessionId').value;
    
    fetch('/clear-history', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json' 
        },
        body: JSON.stringify({ 
            session_id: sessionId
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            console.log('Chat history cleared from backend');
        } else {
            console.error('Error clearing chat history:', data.message);
        }
    })
    .catch(error => {
        console.error('Error clearing chat history:', error);
    });
}

// Clear chat history
function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        chatBox.innerHTML = '';
        localStorage.removeItem('chatHistory');
        emptyState.style.display = 'flex';
        
        // Clear from backend as well
        clearChatHistoryFromBackend();
    }
}

// Add message to chat
function addMessage(text, sender, customTime = null, scroll = true) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.innerHTML = text.replace(/\n/g, '<br>');
    
    const timeSpan = document.createElement('div');
    timeSpan.className = 'timestamp';
    timeSpan.textContent = customTime || getTimestamp();
    messageDiv.appendChild(timeSpan);
    
    chatBox.appendChild(messageDiv);
    
    if (scroll) {
        scrollToBottom();
    }
}

// Show typing indicator
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typingIndicator';
    
    const useWebSearch = toggleWebSearch();
    const statusText = useWebSearch ? 'Searching and analyzing' : 'Thinking';
    
    typingDiv.innerHTML = `
        <span>${statusText}</span>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    chatBox.appendChild(typingDiv);
    scrollToBottom();
}

// Remove typing indicator
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Scroll chat to bottom
function scrollToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Handle Enter key press
function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestionStream();
    }
}

// Clear input field
function clearInput() {
    questionInput.value = '';
    questionInput.focus();
}

// Insert suggestion into input
function insertSuggestion(text) {
    questionInput.value = text;
    questionInput.focus();
}

// Insert sample question
function insertSampleQuestion() {
    const useWebSearch = toggleWebSearch();
    let samples;
    
    if (useWebSearch) {
        samples = [
            "What's the latest news about AI?",
            "Explain quantum computing in simple terms",
            "Best programming languages to learn in 2025",
            "How does photosynthesis work?",
            "What are the health benefits of meditation?",
            "Latest developments in renewable energy"
        ];
    } else {
        samples = [
            "Explain the theory of relativity",
            "What are the principles of good UX design?",
            "How does machine learning work?",
            "What is the capital of France?",
            "Tell me about the history of the internet",
            "What are the benefits of regular exercise?"
        ];
    }
    
    const randomQuestion = samples[Math.floor(Math.random() * samples.length)];
    questionInput.value = randomQuestion;
    questionInput.focus();
}

// RAG Modal Functions
function toggleRagModal() {
    document.getElementById('ragModal').style.display = 'block';
}

function closeRagModal() {
    document.getElementById('ragModal').style.display = 'none';
    // Clear file input and status
    document.getElementById('fileUpload').value = '';
    document.getElementById('fileName').textContent = '';
    document.getElementById('urlInput').value = '';
    document.getElementById('ragStatus').textContent = '';
    document.getElementById('processButton').disabled = true;
}

// File upload handling
document.getElementById('fileUpload').addEventListener('change', function(e) {
    if (this.files && this.files[0]) {
        const file = this.files[0];
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('processButton').disabled = false;
        document.getElementById('ragStatus').textContent = 'File selected. Click "Process Document" to continue.';
        document.getElementById('ragStatus').style.color = 'var(--text-muted)';
    }
});

function processSelectedFile() {
    const fileInput = document.getElementById('fileUpload');
    if (fileInput.files && fileInput.files[0]) {
        uploadFileToRag(fileInput.files[0]);
    } else {
        document.getElementById('ragStatus').textContent = 'Please select a file first.';
        document.getElementById('ragStatus').style.color = 'var(--danger)';
    }
}

function uploadFileToRag(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const status = document.getElementById('ragStatus');
    status.textContent = `Uploading ${file.name}...`;
    status.style.color = 'var(--text-muted)';
    
    fetch('/rag/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            status.textContent = `File "${file.name}" processed successfully!`;
            status.style.color = 'var(--success)';
            
            // Add to processed files list
            addProcessedFile(file.name, 'file');
            
            // Update suggestions
            updateSuggestionsForDocument(file.name);
            
            // Reset file input
            document.getElementById('fileUpload').value = '';
            document.getElementById('fileName').textContent = '';
            document.getElementById('processButton').disabled = true;
        } else {
            status.textContent = 'Error: ' + (data.message || 'Upload failed');
            status.style.color = 'var(--danger)';
        }
    })
    .catch(error => {
        status.textContent = 'Error uploading file';
        status.style.color = 'var(--danger)';
        console.error('Error:', error);
    });
}

function processUrl() {
    const urlInput = document.getElementById('urlInput');
    const url = urlInput.value.trim();
    
    if (!url) {
        document.getElementById('ragStatus').textContent = 'Please enter a URL.';
        document.getElementById('ragStatus').style.color = 'var(--danger)';
        return;
    }
    
    const status = document.getElementById('ragStatus');
    status.textContent = `Processing URL: ${url}...`;
    status.style.color = 'var(--text-muted)';
    
    fetch('/rag/url', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json' 
        },
        body: JSON.stringify({ 
            url: url
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            status.textContent = `URL content processed successfully!`;
            status.style.color = 'var(--success)';
            
            // Add to processed files list
            addProcessedFile(url, 'url');
            
            // Update suggestions
            updateSuggestionsForDocument(url);
            
            // Clear URL input
            urlInput.value = '';
        } else {
            status.textContent = 'Error: ' + (data.message || 'URL processing failed');
            status.style.color = 'var(--danger)';
        }
    })
    .catch(error => {
        status.textContent = 'Error processing URL';
        status.style.color = 'var(--danger)';
        console.error('Error:', error);
    });
}

function addProcessedFile(name, type) {
    const processedFilesList = document.getElementById('processedFilesList');
    const fileElement = document.createElement('div');
    fileElement.className = 'processed-file';
    
    const extension = type === 'url' ? 'URL' : name.split('.').pop().toUpperCase();
    
    fileElement.innerHTML = `
        <div class="file-info">
            <i class="fas ${type === 'url' ? 'fa-link' : 'fa-file'}"></i>
            <span class="file-name">${name.length > 30 ? name.substring(0, 30) + '...' : name}</span>
            <span class="file-type">${extension}</span>
        </div>
        <button class="remove-file" onclick="removeProcessedFile(this)">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    processedFilesList.appendChild(fileElement);
}

function removeProcessedFile(button) {
    const fileElement = button.parentElement;
    fileElement.remove();
    
    // Clear RAG system when all files are removed
    const processedFilesList = document.getElementById('processedFilesList');
    if (processedFilesList.children.length === 0) {
        // Reset to default suggestions
        const suggestions = document.querySelector('.suggestions');
        if (suggestions) {
            suggestions.innerHTML = `
                <div class="suggestion-chip" onclick="insertSuggestion('What is the latest news about AI?')">AI News</div>
                <div class="suggestion-chip" onclick="insertSuggestion('Explain quantum computing')">Quantum Computing</div>
                <div class="suggestion-chip" onclick="insertSuggestion('Best programming languages in 2025')">Programming</div>
            `;
        }
    }
}

function updateSuggestionsForDocument(filename) {
    const suggestions = document.querySelector('.suggestions');
    if (suggestions) {
        const ext = filename.split('.').pop().toLowerCase();
        
        let documentSuggestions = '';
        if (['csv', 'xlsx', 'xls'].includes(ext)) {
            documentSuggestions = `
                <div class="suggestion-chip" onclick="insertSuggestion('Summarize the data in this file')">Summarize Data</div>
                <div class="suggestion-chip" onclick="insertSuggestion('What are the key insights from this data?')">Key Insights</div>
                <div class="suggestion-chip" onclick="insertSuggestion('Show me trends in this data')">Data Trends</div>
            `;
        } else if (['doc', 'docx', 'pdf', 'txt'].includes(ext)) {
            documentSuggestions = `
                <div class="suggestion-chip" onclick="insertSuggestion('Summarize this document')">Summarize</div>
                <div class="suggestion-chip" onclick="insertSuggestion('What are the main points?')">Main Points</div>
                <div class="suggestion-chip" onclick="insertSuggestion('Extract key insights')">Key Insights</div>
            `;
        } else if (['json'].includes(ext)) {
            documentSuggestions = `
                <div class="suggestion-chip" onclick="insertSuggestion('Analyze this JSON data')">Analyze JSON</div>
                <div class="suggestion-chip" onclick="insertSuggestion('What is the structure of this data?')">Data Structure</div>
                <div class="suggestion-chip" onclick="insertSuggestion('Extract key information')">Extract Info</div>
            `;
        } else {
            documentSuggestions = `
                <div class="suggestion-chip" onclick="insertSuggestion('Summarize this content')">Summarize</div>
                <div class="suggestion-chip" onclick="insertSuggestion('What are the key points?')">Key Points</div>
                <div class="suggestion-chip" onclick="insertSuggestion('Analyze the main themes')">Analyze Themes</div>
            `;
        }
        
        suggestions.innerHTML = documentSuggestions;
    }
}

// Close modals when clicking outside
window.onclick = function(event) {
    const mcpModal = document.getElementById('mcpModal');
    const ragModal = document.getElementById('ragModal');
    
    if (event.target === mcpModal) {
        closeMcpModal();
    }
    if (event.target === ragModal) {
        closeRagModal();
    }
}

// Initialize the chat
window.onload = function() {
    loadChatHistory();
    questionInput.focus();
    
    // Initialize web search toggle status
    toggleWebSearch();
    
    // Add event listener to web search toggle
    const webSearchToggle = document.getElementById('webSearchToggle');
    if (webSearchToggle) {
        webSearchToggle.addEventListener('change', toggleWebSearch);
    }
};

// Close modals when clicking outside
window.onclick = function(event) {
    const mcpModal = document.getElementById('mcpModal');
    const ragModal = document.getElementById('ragModal');
    
    if (event.target === mcpModal) {
        closeMcpModal();
    }
    if (event.target === ragModal) {
        closeRagModal();
    }
}

// Initialize the chat
window.onload = function() {
    loadChatHistory();
    questionInput.focus();
    
    // Initialize web search toggle status
    toggleWebSearch();
    
    // Add event listener to web search toggle
    const webSearchToggle = document.getElementById('webSearchToggle');
    if (webSearchToggle) {
        webSearchToggle.addEventListener('change', toggleWebSearch);
    }
};

// MCP Modal Functions
function toggleMcpPanel() {
    document.getElementById('mcpModal').style.display = 'block';
    loadMcpTools();
}

function closeMcpModal() {
    document.getElementById('mcpModal').style.display = 'none';
}

function loadMcpTools() {
    fetch('/mcp/servers')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                const toolsList = document.getElementById('mcpToolsList');
                toolsList.innerHTML = '';
                
                for (const [key, tool] of Object.entries(data.servers)) {
                    if (tool.enabled) {
                        const toolElement = document.createElement('div');
                        toolElement.className = 'mcp-tool';
                        toolElement.innerHTML = `
                            <h4>${tool.name}</h4>
                            <p>${tool.description}</p>
                        `;
                        toolsList.appendChild(toolElement);
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error loading MCP tools:', error);
        });
}

// New function to handle streaming responses
function askQuestionStream() {
    const question = questionInput.value.trim();
    const sessionId = document.getElementById('sessionId').value;
    const useWebSearch = toggleWebSearch();
    
    if (!question) return;

    // Add user message to chat
    addMessage(question, 'user');
    questionInput.value = '';
    
    // Hide empty state if it's the first message
    if (emptyState.style.display !== 'none') {
        emptyState.style.display = 'none';
    }

    // Create a new message element for the bot's response
    const messageId = 'msg-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    messageDiv.id = messageId;
    
    const timeSpan = document.createElement('div');
    timeSpan.className = 'timestamp';
    timeSpan.textContent = getTimestamp();
    
    messageDiv.appendChild(timeSpan);
    chatBox.appendChild(messageDiv);
    
    // Show searching indicator
    showTypingIndicator();
    
    // Scroll to bottom
    scrollToBottom();
    
    // Create EventSource for streaming
    const eventSource = new EventSource(`/ask-stream?question=${encodeURIComponent(question)}&session_id=${sessionId}&web_search=${useWebSearch}`);
    let fullResponse = '';
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        
        if (data.token) {
            fullResponse += data.token;
            document.getElementById(messageId).innerHTML = fullResponse.replace(/\n/g, '<br>') + timeSpan.outerHTML;
            scrollToBottom();
        } else if (data.error) {
            document.getElementById(messageId).innerHTML = `Error: ${data.error}` + timeSpan.outerHTML;
            eventSource.close();
            removeTypingIndicator();
        } else if (data.complete) {
            eventSource.close();
            removeTypingIndicator();
            
            // Save to chat history
            saveChatHistory();
        }
    };
    
    eventSource.onerror = function() {
        eventSource.close();
        removeTypingIndicator();
        // Save to chat history
        saveChatHistory();
    };
}

// Initialize the chat
window.onload = function() {
    loadChatHistory();
    questionInput.focus();
    
    // Initialize web search toggle status
    toggleWebSearch();
    
    // Add event listener to web search toggle
    const webSearchToggle = document.getElementById('webSearchToggle');
    if (webSearchToggle) {
        webSearchToggle.addEventListener('change', toggleWebSearch);
    }
    
    // Close modal when clicking outside
    window.onclick = function(event) {
        const modal = document.getElementById('mcpModal');
        if (event.target === modal) {
            closeMcpModal();
        }
    }
};

// Add to your existing JavaScript

function clearUploads() {
    if (confirm('Are you sure you want to clear all uploaded files? This cannot be undone.')) {
        fetch('/clear-uploads', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json' 
            }
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                alert('Uploads directory cleared successfully');
                // Clear the processed files list
                document.getElementById('processedFilesList').innerHTML = '';
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            alert('Error clearing uploads');
            console.error('Error:', error);
        });
    }
}

function clearChromaDB() {
    if (confirm('Are you sure you want to clear the vector database? All document embeddings will be removed.')) {
        fetch('/clear-chroma', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json' 
            }
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                alert('ChromaDB cleared successfully');
                // Reset to default suggestions
                const suggestions = document.querySelector('.suggestions');
                if (suggestions) {
                    suggestions.innerHTML = `
                        <div class="suggestion-chip" onclick="insertSuggestion('What is the latest news about AI?')">AI News</div>
                        <div class="suggestion-chip" onclick="insertSuggestion('Explain quantum computing')">Quantum Computing</div>
                        <div class="suggestion-chip" onclick="insertSuggestion('Best programming languages in 2025')">Programming</div>
                    `;
                }
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            alert('Error clearing ChromaDB');
            console.error('Error:', error);
        });
    }
}

function clearAllData() {
    if (confirm('Are you sure you want to clear ALL data? This will remove all uploaded files and vector database content.')) {
        fetch('/clear-all', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json' 
            }
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                alert('All data cleared successfully');
                // Clear the processed files list
                document.getElementById('processedFilesList').innerHTML = '';
                // Reset to default suggestions
                const suggestions = document.querySelector('.suggestions');
                if (suggestions) {
                    suggestions.innerHTML = `
                        <div class="suggestion-chip" onclick="insertSuggestion('What is the latest news about AI?')">AI News</div>
                        <div class="suggestion-chip" onclick="insertSuggestion('Explain quantum computing')">Quantum Computing</div>
                        <div class="suggestion-chip" onclick="insertSuggestion('Best programming languages in 2025')">Programming</div>
                    `;
                }
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            alert('Error clearing all data');
            console.error('Error:', error);
        });
    }
}