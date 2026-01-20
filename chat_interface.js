// Configure PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

class CADChatInterface {
    constructor() {
        this.pdfFile = null;
        this.pdfDoc = null;
        this.currentPage = 1;
        this.totalPages = 0;
        this.scale = 1.5;
        this.chatHistory = [];
        this.isProcessing = false;
        this.sessionId = null;
        this.documentId = null;
        // Use current hostname and port for API calls
        this.apiUrl = `${window.location.protocol}//${window.location.host}/api`;
        
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        // Upload elements
        this.uploadScreen = document.getElementById('uploadScreen');
        this.dropZone = document.getElementById('dropZone');
        this.fileInput = document.getElementById('fileInput');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileName = document.getElementById('fileName');
        this.uploadBtn = document.getElementById('uploadBtn');

        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.chatStatus = document.getElementById('chatStatus');

        // PDF elements
        this.pdfViewerContainer = document.getElementById('pdfViewerContainer');
        this.pdfFileName = document.getElementById('pdfFileName');
        this.currentPageSpan = document.getElementById('currentPage');
        this.totalPagesSpan = document.getElementById('totalPages');
        this.prevPageBtn = document.getElementById('prevPage');
        this.nextPageBtn = document.getElementById('nextPage');
        this.zoomInBtn = document.getElementById('zoomIn');
        this.zoomOutBtn = document.getElementById('zoomOut');
        this.downloadBtn = document.getElementById('downloadPdf');
        
        // Header action buttons
        this.clearBtn = document.querySelector('[data-tooltip="Clear Chat"]');
        this.saveBtn = document.querySelector('[data-tooltip="Save History"]');
        
        // Image Search elements
        this.imageSearchBtn = document.getElementById('imageSearchBtn');
        this.imageSearchPanel = document.getElementById('imageSearchPanel');
        this.panelOverlay = document.getElementById('panelOverlay');
        this.closePanelBtn = document.getElementById('closePanelBtn');
        this.imageSearchInput = document.getElementById('imageSearchInput');
        this.textSearchBtn = document.getElementById('textSearchBtn');
        this.imageUploadZone = document.getElementById('imageUploadZone');
        this.imageSearchFile = document.getElementById('imageSearchFile');
        this.searchResults = document.getElementById('searchResults');
        
        // Image Preview Modal
        this.imagePreviewModal = document.getElementById('imagePreviewModal');
        this.previewImage = document.getElementById('previewImage');
        this.previewInfo = document.getElementById('previewInfo');
        this.closePreviewBtn = document.getElementById('closePreviewBtn');
    }

    attachEventListeners() {
        // File upload listeners
        this.dropZone.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.uploadBtn.addEventListener('click', () => this.startChatSession());

        // Drag and drop
        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.classList.add('drag-over');
        });

        this.dropZone.addEventListener('dragleave', () => {
            this.dropZone.classList.remove('drag-over');
        });

        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type === 'application/pdf') {
                this.pdfFile = files[0];
                this.updateFileInfo();
            }
        });

        // Chat listeners
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // PDF navigation listeners
        this.prevPageBtn.addEventListener('click', () => this.goToPreviousPage());
        this.nextPageBtn.addEventListener('click', () => this.goToNextPage());
        this.zoomInBtn.addEventListener('click', () => this.zoomIn());
        this.zoomOutBtn.addEventListener('click', () => this.zoomOut());
        this.downloadBtn.addEventListener('click', () => this.downloadPDF());
        
        // Header action listeners
        this.clearBtn.addEventListener('click', () => this.clearChat());
        this.saveBtn.addEventListener('click', () => this.saveHistory());
        
        // Image Search Panel listeners
        this.imageSearchBtn.addEventListener('click', () => this.toggleImageSearchPanel());
        this.closePanelBtn.addEventListener('click', () => this.closeImageSearchPanel());
        this.panelOverlay.addEventListener('click', () => this.closeImageSearchPanel());
        
        // Text search
        this.textSearchBtn.addEventListener('click', () => this.searchByText());
        this.imageSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.searchByText();
            }
        });
        
        // Image upload search
        this.imageUploadZone.addEventListener('click', () => this.imageSearchFile.click());
        this.imageSearchFile.addEventListener('change', (e) => this.searchByImage(e));
        
        this.imageUploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.imageUploadZone.classList.add('drag-over');
        });
        
        this.imageUploadZone.addEventListener('dragleave', () => {
            this.imageUploadZone.classList.remove('drag-over');
        });
        
        this.imageUploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.imageUploadZone.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                this.searchByImageFile(files[0]);
            }
        });
        
        // Image preview modal
        this.closePreviewBtn.addEventListener('click', () => this.closeImagePreview());
        this.imagePreviewModal.addEventListener('click', (e) => {
            if (e.target === this.imagePreviewModal) {
                this.closeImagePreview();
            }
        });
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file && file.type === 'application/pdf') {
            this.pdfFile = file;
            this.updateFileInfo();
        }
    }

    updateFileInfo() {
        this.fileName.textContent = this.pdfFile.name;
        this.fileInfo.classList.add('show');
        this.uploadBtn.disabled = false;
    }

    async startChatSession() {
        if (!this.pdfFile) return;

        this.chatStatus.textContent = 'Uploading document...';
        this.uploadBtn.disabled = true;
        this.uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin" style="margin-right: 8px;"></i>Uploading...';

        try {
            // Upload PDF to backend
            const formData = new FormData();
            formData.append('file', this.pdfFile);

            const response = await fetch(`${this.apiUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Upload failed');
            }

            this.sessionId = data.session_id;
            this.documentId = data.document_id;

            // Hide upload screen with animation
            this.uploadScreen.classList.add('hidden');

            // Load PDF for viewing
            await this.loadPDF();

            // Enable chat
            this.chatInput.disabled = false;
            this.sendBtn.disabled = false;
            this.chatStatus.textContent = 'Analyzing document...';

            // Clear empty state
            this.chatMessages.innerHTML = '';

            // Add welcome message
            setTimeout(() => {
                this.addAssistantMessage(
                    `Hello! I've loaded **${this.pdfFile.name}** (${this.totalPages} pages). I'm ready to help you analyze this CAD document. You can ask me about:\n\n` +
                    `• Drawing numbers and specifications\n` +
                    `• Room dimensions and layouts\n` +
                    `• Symbol counts (doors, windows, fixtures)\n` +
                    `• Technical annotations\n` +
                    `• Spatial analysis\n\n` +
                    `🔍 **Image Search** is ready! Click the image button to search similar drawings.\n\n` +
                    `Try asking: "List all drawing numbers in this document" or "How many doors on the 1st floor?"`
                );
                this.chatStatus.textContent = 'Ready';
            }, 500);

        } catch (error) {
            console.error('Upload error:', error);
            alert(`Failed to upload PDF: ${error.message}`);
            this.uploadBtn.disabled = false;
            this.uploadBtn.innerHTML = '<i class="fas fa-rocket" style="margin-right: 8px;"></i>Start Chat Session';
        }
    }

    async loadPDF() {
        try {
            const fileReader = new FileReader();
            
            return new Promise((resolve, reject) => {
                fileReader.onload = async (e) => {
                    const typedArray = new Uint8Array(e.target.result);
                    
                    try {
                        this.pdfDoc = await pdfjsLib.getDocument(typedArray).promise;
                        this.totalPages = this.pdfDoc.numPages;
                        this.currentPage = 1;

                        // Update UI
                        this.pdfFileName.textContent = this.pdfFile.name;
                        this.currentPageSpan.textContent = this.currentPage;
                        this.totalPagesSpan.textContent = this.totalPages;

                        // Enable controls
                        this.prevPageBtn.disabled = false;
                        this.nextPageBtn.disabled = false;
                        this.zoomInBtn.disabled = false;
                        this.zoomOutBtn.disabled = false;
                        this.downloadBtn.disabled = false;

                        // Render first page
                        await this.renderPage(this.currentPage);
                        resolve();
                    } catch (error) {
                        reject(error);
                    }
                };

                fileReader.onerror = reject;
                fileReader.readAsArrayBuffer(this.pdfFile);
            });
        } catch (error) {
            console.error('Error loading PDF:', error);
            alert('Failed to load PDF. Please try again.');
        }
    }

    async renderPage(pageNum) {
        try {
            const page = await this.pdfDoc.getPage(pageNum);
            const viewport = page.getViewport({ scale: this.scale });

            // Create or get canvas
            let canvas = document.getElementById('pdfCanvas');
            if (!canvas) {
                canvas = document.createElement('canvas');
                canvas.id = 'pdfCanvas';
                this.pdfViewerContainer.innerHTML = '';
                this.pdfViewerContainer.appendChild(canvas);
            }

            const context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            const renderContext = {
                canvasContext: context,
                viewport: viewport
            };

            await page.render(renderContext).promise;
            
            // Update page info
            this.currentPageSpan.textContent = this.currentPage;
            
            // Update button states
            this.prevPageBtn.disabled = this.currentPage === 1;
            this.nextPageBtn.disabled = this.currentPage === this.totalPages;

        } catch (error) {
            console.error('Error rendering page:', error);
        }
    }

    async goToPage(pageNum) {
        if (pageNum < 1 || pageNum > this.totalPages) return;
        
        this.currentPage = pageNum;
        await this.renderPage(this.currentPage);
        
        // Smooth scroll to top
        this.pdfViewerContainer.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    }

    async goToPreviousPage() {
        if (this.currentPage > 1) {
            await this.goToPage(this.currentPage - 1);
        }
    }

    async goToNextPage() {
        if (this.currentPage < this.totalPages) {
            await this.goToPage(this.currentPage + 1);
        }
    }

    async zoomIn() {
        this.scale = Math.min(this.scale + 0.25, 3);
        await this.renderPage(this.currentPage);
    }

    async zoomOut() {
        this.scale = Math.max(this.scale - 0.25, 0.5);
        await this.renderPage(this.currentPage);
    }

    downloadPDF() {
        const url = URL.createObjectURL(this.pdfFile);
        const a = document.createElement('a');
        a.href = url;
        a.download = this.pdfFile.name;
        a.click();
        URL.revokeObjectURL(url);
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isProcessing || !this.sessionId) return;

        // Add user message
        this.addUserMessage(message);
        this.chatInput.value = '';
        this.isProcessing = true;
        this.sendBtn.disabled = true;
        this.chatStatus.textContent = 'Thinking...';

        // Show typing indicator
        this.addTypingIndicator();

        try {
            // Make API call to backend
            const response = await fetch(`${this.apiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    message: message
                })
            });

            const data = await response.json();

            this.removeTypingIndicator();

            if (data.success) {
                this.addAssistantMessage(data.answer);
                this.chatStatus.textContent = 'Ready';
            } else {
                this.addAssistantMessage(`❌ Error: ${data.error}`);
                this.chatStatus.textContent = 'Error occurred';
            }

        } catch (error) {
            console.error('Chat error:', error);
            this.removeTypingIndicator();
            this.addAssistantMessage(`❌ Failed to get response: ${error.message}`);
            this.chatStatus.textContent = 'Error occurred';
        } finally {
            this.isProcessing = false;
            this.sendBtn.disabled = false;
        }
    }

    addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-user"></i>
            </div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(text)}</div>
            </div>
        `;
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addAssistantMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        
        // Process text for citations and markdown
        const processedText = this.processMessageText(text);
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-text">${processedText}</div>
            </div>
        `;
        this.chatMessages.appendChild(messageDiv);
        
        // Attach citation click handlers
        messageDiv.querySelectorAll('.citation').forEach(citation => {
            citation.addEventListener('click', () => {
                const page = parseInt(citation.dataset.page);
                this.goToPage(page);
                this.highlightPDFPanel();
            });
        });
        
        this.scrollToBottom();
    }

    processMessageText(text) {
        // Escape HTML first to prevent XSS
        text = text.replace(/&/g, '&amp;')
                   .replace(/</g, '&lt;')
                   .replace(/>/g, '&gt;')
                   .replace(/"/g, '&quot;')
                   .replace(/'/g, '&#039;');
        
        // Convert markdown-style bold **text**
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convert markdown-style italic *text*
        text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        
        // Convert markdown-style code `code`
        text = text.replace(/`([^`]+)`/g, '<code style="background: var(--surface-light); padding: 2px 6px; border-radius: 4px; font-family: monospace;">$1</code>');
        
        // Add citations [page X] or [pages X-Y] -> clickable citation
        text = text.replace(/\[pages? (\d+)(?:-(\d+))?\]/gi, (match, page1, page2) => {
            if (page2) {
                return `<span class="citation" data-page="${page1}">
                    <i class="fas fa-file-alt"></i> Pages ${page1}-${page2}
                </span>`;
            } else {
                return `<span class="citation" data-page="${page1}">
                    <i class="fas fa-file-alt"></i> Page ${page1}
                </span>`;
            }
        });
        
        // Handle bullet points with • character
        text = text.replace(/^([•\-\*])\s+(.+)$/gm, '<div style="margin-left: 20px;">$1 $2</div>');
        
        // Handle numbered lists
        text = text.replace(/^(\d+)\.\s+(.+)$/gm, '<div style="margin-left: 20px;">$1. $2</div>');
        
        // Convert double newlines to paragraph breaks
        text = text.replace(/\n\n/g, '<br><br>');
        
        // Convert single newlines to breaks
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }

    addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = 'typingIndicator';
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    highlightPDFPanel() {
        // Add visual feedback when navigating to citation
        const pdfPanel = document.querySelector('.pdf-panel');
        pdfPanel.style.boxShadow = '0 0 0 3px var(--primary)';
        setTimeout(() => {
            pdfPanel.style.boxShadow = '';
        }, 1000);
    }

    scrollToBottom() {
        this.chatMessages.scrollTo({
            top: this.chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }

    async clearChat() {
        if (!this.sessionId) return;
        
        if (!confirm('Are you sure you want to clear the chat history?')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiUrl}/clear/${this.sessionId}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                this.chatMessages.innerHTML = '';
                this.chatHistory = [];
                this.addAssistantMessage(
                    `Chat history cleared! I'm ready to answer new questions about **${this.pdfFile.name}**.`
                );
                this.chatStatus.textContent = 'History cleared';
            } else {
                alert(`Failed to clear history: ${data.error}`);
            }

        } catch (error) {
            console.error('Clear error:', error);
            alert(`Failed to clear history: ${error.message}`);
        }
    }

    async saveHistory() {
        if (!this.sessionId) return;

        try {
            this.chatStatus.textContent = 'Saving history...';

            const response = await fetch(`${this.apiUrl}/save/${this.sessionId}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                alert(`Chat history saved successfully to:\n${data.filename}`);
                this.chatStatus.textContent = 'History saved';
            } else {
                alert(`Failed to save history: ${data.error}`);
                this.chatStatus.textContent = 'Save failed';
            }

        } catch (error) {
            console.error('Save error:', error);
            alert(`Failed to save history: ${error.message}`);
            this.chatStatus.textContent = 'Save failed';
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ==================== IMAGE SEARCH METHODS ====================

    toggleImageSearchPanel() {
        this.imageSearchPanel.classList.toggle('open');
        this.panelOverlay.classList.toggle('active');
        this.imageSearchBtn.classList.toggle('active');
    }

    closeImageSearchPanel() {
        this.imageSearchPanel.classList.remove('open');
        this.panelOverlay.classList.remove('active');
        this.imageSearchBtn.classList.remove('active');
    }

    async searchByText() {
        const query = this.imageSearchInput.value.trim();
        if (!query) return;

        this.textSearchBtn.disabled = true;
        this.textSearchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        this.showSearchLoading();

        try {
            const response = await fetch(`${this.apiUrl}/image-search/text`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    document_id: this.documentId,
                    limit: 10
                })
            });

            const data = await response.json();

            if (data.success) {
                this.displaySearchResults(data.results);
            } else {
                this.showSearchError(data.error);
            }

        } catch (error) {
            console.error('Search error:', error);
            this.showSearchError(error.message);
        } finally {
            this.textSearchBtn.disabled = false;
            this.textSearchBtn.innerHTML = '<i class="fas fa-search"></i>';
        }
    }

    searchByImage(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            this.searchByImageFile(file);
        }
    }

    async searchByImageFile(file) {
        this.showSearchLoading();

        try {
            const formData = new FormData();
            formData.append('image', file);
            if (this.documentId) {
                formData.append('document_id', this.documentId);
            }
            formData.append('limit', '10');

            const response = await fetch(`${this.apiUrl}/image-search/image`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.displaySearchResults(data.results);
            } else {
                this.showSearchError(data.error);
            }

        } catch (error) {
            console.error('Image search error:', error);
            this.showSearchError(error.message);
        }
    }

    showSearchLoading() {
        this.searchResults.innerHTML = `
            <div class="no-results">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Searching...</p>
            </div>
        `;
    }

    showSearchError(message) {
        this.searchResults.innerHTML = `
            <div class="no-results">
                <i class="fas fa-exclamation-triangle" style="color: var(--error);"></i>
                <p>${message}</p>
            </div>
        `;
    }

    displaySearchResults(results) {
        if (!results || results.length === 0) {
            this.searchResults.innerHTML = `
                <div class="no-results">
                    <i class="fas fa-search"></i>
                    <p>No matching images found</p>
                </div>
            `;
            return;
        }

        this.searchResults.innerHTML = results.map(result => `
            <div class="search-result-item" data-page="${result.page_number}" data-image-id="${result.image_id}">
                ${result.thumbnail ? 
                    `<img class="result-thumbnail" src="data:image/png;base64,${result.thumbnail}" alt="Page ${result.page_number}">` :
                    `<div class="result-thumbnail" style="display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-file-image" style="font-size: 32px; color: var(--text-muted);"></i>
                    </div>`
                }
                <div class="result-info">
                    <div class="result-title">
                        <span>${result.document_name || 'Document'}</span>
                        <span class="result-page-badge">Page ${result.page_number}</span>
                    </div>
                    <div class="result-description">${result.description || 'No description available'}</div>
                    <div class="result-score">
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${Math.round(result.score * 100)}%"></div>
                        </div>
                        <span class="score-value">${Math.round(result.score * 100)}%</span>
                    </div>
                    ${result.image_type ? `
                        <div class="result-tags">
                            <span class="result-tag">${result.image_type}</span>
                        </div>
                    ` : ''}
                </div>
            </div>
        `).join('');

        // Attach click handlers to results
        this.searchResults.querySelectorAll('.search-result-item').forEach(item => {
            item.addEventListener('click', () => {
                const page = parseInt(item.dataset.page);
                this.goToPage(page);
                this.highlightPDFPanel();
                this.closeImageSearchPanel();
            });
        });
    }

    openImagePreview(imageId, thumbnail, info) {
        this.previewImage.src = thumbnail ? `data:image/png;base64,${thumbnail}` : '';
        this.previewInfo.innerHTML = `
            <p><strong>Page:</strong> ${info.page_number}</p>
            <p><strong>Type:</strong> ${info.image_type || 'Unknown'}</p>
            <p><strong>Description:</strong> ${info.description || 'No description'}</p>
        `;
        this.imagePreviewModal.classList.add('active');
    }

    closeImagePreview() {
        this.imagePreviewModal.classList.remove('active');
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new CADChatInterface();
});
