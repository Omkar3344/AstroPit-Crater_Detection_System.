document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded");
    
    // Force initial scroll to top without animation
    window.scrollTo(0, 0);
    
    // Add a delay before applying smooth scroll to hero section
    setTimeout(() => {
        // Use smooth scrolling behavior for better UX
        const heroSection = document.getElementById('hero');
        if (heroSection) {
            heroSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }, 300); // Slightly longer delay for smoother effect
    
    // Cache DOM elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const resultStatus = document.getElementById('result-status');
    const resultMessage = document.getElementById('result-message');
    const resultsSection = document.getElementById('results');
    const imageResults = document.getElementById('image-results');
    const scanningOverlay = document.querySelector('.scanning-overlay');
    
    // Image elements
    const originalImage = document.getElementById('original-image');
    const resultImage = document.getElementById('result-image');
    const downloadOriginal = document.getElementById('download-original');
    const downloadResult = document.getElementById('download-result');
    
    // Add this near the start of your script.js file
    function updateDebugInfo(data) {
        const debugData = document.getElementById('debug-data');
        if (debugData) {
            debugData.textContent = JSON.stringify(data, null, 2);
        }
    }
    
    // Define the elements
    window.originalImage = originalImage;
    window.resultImage = resultImage;
    window.downloadOriginal = downloadOriginal;
    window.downloadResult = downloadResult;
    
    // Result containers
    window.imageResults = imageResults;
    window.resultStatus = resultStatus;
    window.resultMessage = resultMessage;
    window.resultsSection = resultsSection;
    window.scanningOverlay = scanningOverlay;
    
    // Get all required elements
    const browseBtn = document.getElementById('browse-btn');
    const filePreview = document.getElementById('file-preview');
    
    const newDetectionBtn = document.getElementById('new-detection-btn');
    
    // Log which elements were found
    console.log("Elements found:", {
        uploadArea: !!uploadArea,
        fileInput: !!fileInput,
        browseBtn: !!browseBtn,
        uploadBtn: !!uploadBtn
    });
    
    // Browse button click handler
    if (browseBtn && fileInput) {
        console.log("Adding click handler to browse button");
        browseBtn.addEventListener('click', function(e) {
            console.log("Browse button clicked");
            fileInput.click();
        });
    }
    
    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            console.log("File input changed");
            if (fileInput.files.length > 0) {
                // Only allow image files
                const file = fileInput.files[0];
                if (!file.type.startsWith('image/')) {
                    alert('Please select an image file (JPEG, PNG, etc.)');
                    fileInput.value = '';
                    return;
                }
                
                displayFilePreview(file);
                if (uploadBtn) uploadBtn.removeAttribute('disabled');
            }
        });
    }
    
    // Drag and drop handlers
    if (uploadArea) {
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', function() {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length > 0 && fileInput) {
                fileInput.files = e.dataTransfer.files;
                displayFilePreview(e.dataTransfer.files[0]);
                if (uploadBtn) uploadBtn.removeAttribute('disabled');
            }
        });
    }
    
    // Upload button click handler
    if (uploadBtn && fileInput) {
        uploadBtn.addEventListener('click', function() {
            console.log("Upload button clicked");
            if (fileInput.files.length > 0) {
                uploadFile(fileInput.files[0]);
            }
        });
    }
    
    // New detection button handler
    if (newDetectionBtn) {
        newDetectionBtn.addEventListener('click', function() {
            console.log("New detection button clicked");
            if (fileInput) fileInput.value = '';
            if (filePreview) filePreview.innerHTML = '';
            if (uploadBtn) uploadBtn.setAttribute('disabled', 'true');
            if (resultsSection) resultsSection.classList.add('hidden');
            
            // Scroll to upload section
            const uploadSection = document.getElementById('upload');
            if (uploadSection) {
                uploadSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
    
    // File preview display function
    function displayFilePreview(file) {
        console.log("Displaying file preview for:", file.name);
        if (!filePreview) return;
        
        // Clear previous preview
        filePreview.innerHTML = '';
        
        // Create file info element
        const fileInfo = document.createElement('div');
        fileInfo.className = 'file-info';
        
        // Format file size
        const fileSize = formatFileSize(file.size);
        fileInfo.innerHTML = `
            <span class="file-name">${file.name}</span>
            <span class="file-size">${fileSize}</span>
        `;
        filePreview.appendChild(fileInfo);
        
        // Only handle image previews
        if (file.type.startsWith('image/')) {
            const preview = document.createElement('div');
            preview.className = 'preview-image';
            
            const img = document.createElement('img');
            const reader = new FileReader();
            reader.onload = function(e) {
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
            
            preview.appendChild(img);
            filePreview.appendChild(preview);
        }
    }
    
    // Format file size
    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' bytes';
        } else if (bytes < 1048576) {
            return (bytes / 1024).toFixed(1) + ' KB';
        } else {
            return (bytes / 1048576).toFixed(1) + ' MB';
        }
    }
    
    // Simplified uploadFile function that only handles images
    function uploadFile(file) {
        console.log("Starting file upload:", file.name);
        
        // Ensure it's an image file
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPEG, PNG, etc.)');
            return;
        }
        
        // Show scanning overlay
        if (scanningOverlay) {
            scanningOverlay.style.display = 'flex';
        }
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Set timeout for image processing (3 minutes)
        const timeoutDuration = 180000;
        
        // Show timeout information to user
        if (resultMessage) {
            resultMessage.textContent = `Processing image... (timeout: ${Math.round(timeoutDuration/60000)} minutes)`;
        }
        
        // Set a timeout to handle long-running requests
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.log("Request timeout reached, aborting fetch");
            controller.abort();
        }, timeoutDuration);
        
        // Use the fetch API with proper error handling
        fetch('/detect/', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        })
        .then(async response => {
            // Clear timeout as soon as we get a response
            clearTimeout(timeoutId);
            
            console.log("Response status:", response.status);
            
            // Get response as text first to ensure we can debug even non-JSON responses
            const responseText = await response.text();
            console.log("Response text preview:", responseText.substring(0, 200));
            
            // Try to parse as JSON
            try {
                const data = JSON.parse(responseText);
                
                // Check for error response
                if (!response.ok) {
                    throw new Error(data.detail || `Server error: ${response.status}`);
                }
                
                return data;
            } catch (e) {
                console.error("JSON parse error:", e);
                throw new Error(`Failed to parse server response: ${e.message}`);
            }
        })
        .then(data => {
            console.log("Processed response data:", data);
            
            // Hide scanning overlay
            if (scanningOverlay) {
                scanningOverlay.style.display = 'none';
            }
            
            // Show results section
            if (resultsSection) {
                resultsSection.classList.remove('hidden');
            }
            
            // Handle image results
            handleImageResult(data);
            
            // Scroll to results
            if (resultsSection) {
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
        })
        .catch(error => {
            console.error("Error during upload:", error);
            
            // Always clear the timeout
            clearTimeout(timeoutId);
            
            // Hide scanning overlay
            if (scanningOverlay) {
                scanningOverlay.style.display = 'none';
            }
            
            // Show error message
            if (resultStatus && resultMessage) {
                resultStatus.className = 'result-status error';
                resultMessage.textContent = 'Error: ' + (error.message || 'Unknown error during file processing');
            }
            
            // Still show results section but with error
            if (resultsSection) {
                resultsSection.classList.remove('hidden');
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
    
    // Update handleImageResult to ensure visibility
    function handleImageResult(data) {
        console.log("Handling image result with data:", data);
        
        if (!resultStatus || !resultMessage) return;
        
        // Determine if it's a success or warning
        if (data.message && data.message.includes('Not a planetary surface')) {
            resultStatus.className = 'result-status warning';
            resultMessage.textContent = data.message;
        } else {
            resultStatus.className = 'result-status success';
            resultMessage.textContent = data.message || 'Detection completed successfully';
        }
        
        // Make sure image results are visible
        if (imageResults) {
            imageResults.classList.remove('hidden');
            console.log("Image results should be visible now");
        }
        
        // Update images with cache-busting timestamps to ensure they refresh
        const timestamp = new Date().getTime();
        
        // Set image sources
        if (originalImage && data.original_path) {
            originalImage.src = data.original_path + '?t=' + timestamp;
            console.log("Set original image src to:", originalImage.src);
        }
        
        if (resultImage && data.result_path) {
            resultImage.src = data.result_path + '?t=' + timestamp;
            console.log("Set result image src to:", resultImage.src);
        }
        
        // Set download links
        if (downloadOriginal && data.original_download) {
            downloadOriginal.href = data.original_download;
            downloadOriginal.style.display = 'inline-block';
        }
        
        if (downloadResult && data.result_download) {
            downloadResult.href = data.result_download;
            downloadResult.style.display = 'inline-block';
        }
        
        // Update technical data section
        updateTechnicalData(data);
        
        // Make technical data section visible
        document.getElementById('technical-data').classList.remove('hidden');
        
        // Also initialize 3D visualization
        initialize3DVisualization(data);
    }
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Add these new event listeners
    const copyJsonBtn = document.getElementById('copy-json-btn');
    const downloadJsonBtn = document.getElementById('download-json-btn');
    const toggleFormatBtn = document.getElementById('toggle-format-btn');
    
    if (copyJsonBtn) {
        copyJsonBtn.addEventListener('click', function() {
            if (window.craterData) {
                const jsonString = JSON.stringify(window.craterData, null, 2);
                navigator.clipboard.writeText(jsonString).then(function() {
                    alert('JSON data copied to clipboard');
                }).catch(function(err) {
                    alert('Error copying to clipboard: ' + err);
                });
            } else {
                alert('No crater data available to copy');
            }
        });
    }
    
    if (downloadJsonBtn) {
        downloadJsonBtn.addEventListener('click', function() {
            if (window.craterData) {
                const jsonString = JSON.stringify(window.craterData, null, 2);
                const blob = new Blob([jsonString], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                
                const a = document.createElement('a');
                a.href = url;
                a.download = 'crater_detection_data.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            } else {
                alert('No crater data available to download');
            }
        });
    }
    
    if (toggleFormatBtn) {
        toggleFormatBtn.textContent = 'Switch to JSON View';
        toggleFormatBtn.removeEventListener('click', toggleFormatView); // Remove any existing handlers
        toggleFormatBtn.addEventListener('click', toggleFormatView);
    }

    // Optimize scrolling for the JSON container
    const jsonContainer = document.querySelector('.json-container');
    if (jsonContainer) {
        let ticking = false;
        jsonContainer.addEventListener('scroll', function() {
            if (!ticking) {
                window.requestAnimationFrame(function() {
                    // Scroll handling logic (if any)
                    ticking = false;
                });
                ticking = true;
            }
        });
    }

    // Initialize technical data section
    initializeTechnicalDataSection();

    // Add 3D button click handler
    const view3DBtn = document.getElementById('view-3d-btn');
    if (view3DBtn) {
        view3DBtn.addEventListener('click', function() {
            const visualizationSection = document.getElementById('visualization-3d');
            if (visualizationSection) {
                visualizationSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
});

// Toggle between table and JSON views
function toggleFormatView() {
    const jsonContainer = document.getElementById('crater-data-json');
    const toggleFormatBtn = document.getElementById('toggle-format-btn');
    
    if (!jsonContainer || !window.craterData || !toggleFormatBtn) return;
    
    // Check current display mode
    const isTableMode = jsonContainer.classList.contains('table-mode');
    
    if (isTableMode) {
        // Switch to JSON view
        toggleFormatBtn.textContent = 'Switch to Table View';
        jsonContainer.classList.remove('table-mode');
        jsonContainer.classList.add('json-mode');
        
        // Clear current content
        jsonContainer.innerHTML = '<div class="spinner"></div><p>Loading JSON view...</p>';
        
        // Use setTimeout to prevent UI freeze
        setTimeout(() => {
            jsonContainer.innerHTML = formatJSON(window.craterData);
            
            // Optimize large JSON display
            if (JSON.stringify(window.craterData).length > 100000) {
                chunkifyLargeJSON(jsonContainer);
            }
        }, 10);
    } else {
        // Switch to table view
        toggleFormatBtn.textContent = 'Switch to JSON View';
        jsonContainer.classList.add('table-mode');
        jsonContainer.classList.remove('json-mode');
        
        // Clear and show loading
        jsonContainer.innerHTML = '<div class="spinner"></div><p>Loading table view...</p>';
        
        // Use setTimeout to prevent UI freeze
        setTimeout(() => {
            // Generate table HTML
            const tableHTML = convertJSONToTable(window.craterData);
            jsonContainer.innerHTML = tableHTML;
            
            // Setup pagination after rendering
            setupTablePagination(jsonContainer, window.craterData);
        }, 10);
    }
}

// Format JSON with syntax highlighting
function formatJSON(json) {
    if (!json) return 'No data available';
    
    try {
        // Convert to string with indentation
        const jsonString = JSON.stringify(json, null, 2);
        
        // For very large strings, return without syntax highlighting
        if (jsonString.length > 500000) {
            return jsonString;
        }
        
        // Apply syntax highlighting
        return jsonString.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function(match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            return '<span class="' + cls + '">' + match + '</span>';
        });
    } catch (e) {
        console.error("JSON formatting error:", e);
        return `Error formatting JSON: ${e.message}`;
    }
}

// Optimize large JSON display by chunking
function chunkifyLargeJSON(jsonElement) {
    if (!jsonElement || jsonElement.innerHTML.length < 100000) return;

    console.log("Optimizing large JSON display...");
    
    // Get the content
    const content = jsonElement.innerHTML;
    
    // Create document fragment
    const fragment = document.createDocumentFragment();
    
    // Split content into chunks
    const chunkSize = 50000;
    let position = 0;
    
    while (position < content.length) {
        const chunk = document.createElement('span');
        chunk.innerHTML = content.slice(position, position + chunkSize);
        fragment.appendChild(chunk);
        position += chunkSize;
    }
    
    // Clear and replace content
    jsonElement.innerHTML = '';
    jsonElement.appendChild(fragment);
}

// Initialize technical data section
function initializeTechnicalDataSection() {
    // Check if technical data section exists and ensure it's properly set up
    const technicalSection = document.getElementById('technical-data');
    if (!technicalSection) {
        console.warn("Technical data section not found in DOM");
        return;
    }
    
    console.log("Initializing technical data section...");
    
    // Initialize the buttons if they exist
    const copyJsonBtn = document.getElementById('copy-json-btn');
    const downloadJsonBtn = document.getElementById('download-json-btn');
    const toggleFormatBtn = document.getElementById('toggle-format-btn');
    
    // Set initial format button text
    if (toggleFormatBtn) {
        toggleFormatBtn.textContent = 'Switch to JSON View';
    }
    
    // Ensure the crater-data-json container exists
    const jsonContainer = document.getElementById('crater-data-json');
    if (!jsonContainer) {
        console.warn("JSON container not found in technical data section");
    } else {
        // Set default class for table mode
        jsonContainer.classList.add('table-mode');
        jsonContainer.classList.remove('json-mode');
    }
    
    // Check if convertJSONToTable function exists
    if (typeof convertJSONToTable !== 'function') {
        console.error("convertJSONToTable function is not defined");
        
        // Define a simple placeholder if it's missing
        window.convertJSONToTable = function(data) {
            return '<div class="table-placeholder">Table view requires the convertJSONToTable function.</div>';
        };
    }
    
    // Check if setupTablePagination function exists
    if (typeof setupTablePagination !== 'function') {
        console.error("setupTablePagination function is not defined");
        
        // Define a simple placeholder if it's missing
        window.setupTablePagination = function() {
            console.warn("Table pagination not available");
        };
    }
    
    console.log("Technical data section initialized");
}

// Convert JSON to HTML tables
function convertJSONToTable(jsonData) {
    if (!jsonData) return '<div class="empty-table-message">No data available</div>';
    
    // Create container for tables
    let html = '<div class="table-container">';
    
    // 1. Add Summary Table
    html += `
        <table class="data-table summary-table">
            <thead>
                <tr><th colspan="2">Detection Summary</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td>Total Craters</td>
                    <td>${jsonData.summary?.total_craters || 0}</td>
                </tr>
                <tr>
                    <td>Average Confidence</td>
                    <td>${jsonData.summary?.average_confidence || 0}%</td>
                </tr>
                <tr>
                    <td>Average Size</td>
                    <td>Width: ${jsonData.summary?.average_size?.width || 0}px, Height: ${jsonData.summary?.average_size?.height || 0}px</td>
                </tr>
                <tr>
                    <td>Detection Time</td>
                    <td>${jsonData.summary?.detection_time || 'N/A'}</td>
                </tr>
            </tbody>
        </table>`;
    
    // 2. Add Media Info Table
    if (jsonData.media_info) {
        html += `
            <table class="data-table media-table">
                <thead>
                    <tr><th colspan="2">Image Information</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Name</td>
                        <td>${jsonData.media_info.name || 'N/A'}</td>
                    </tr>
                    <tr>
                        <td>Dimensions</td>
                        <td>${jsonData.media_info.dimensions?.width || 0} × ${jsonData.media_info.dimensions?.height || 0}</td>
                    </tr>
                </tbody>
            </table>`;
    }
    
    // 3. Add Craters Table
    const cratersData = jsonData.craters || (Array.isArray(jsonData) ? jsonData : []);
    
    if (cratersData.length > 0) {
        const PAGE_SIZE = 10;
        const totalPages = Math.ceil(cratersData.length / PAGE_SIZE);
        const cratesToShow = cratersData.slice(0, PAGE_SIZE);
        
        html += `
            <div class="table-section">
                <h3>Detected Craters <span class="crater-count">(${cratersData.length} total)</span></h3>
                <div class="craters-table-container">
                    <table class="data-table craters-table" id="craters-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Confidence</th>
                                <th>Size</th>
                                <th>Position</th>
                                <th>Depth</th>
                            </tr>
                        </thead>
                        <tbody>`;
        
        // Add first page of crater data
        cratesToShow.forEach(crater => {
            html += `
                <tr>
                    <td>${crater.id || 'N/A'}</td>
                    <td>${crater.confidence || 0}%</td>
                    <td>${crater.bounding_box?.width || 0} × ${crater.bounding_box?.height || 0}</td>
                    <td>Center: (${crater.center?.x || 0}, ${crater.center?.y || 0})</td>
                    <td>${crater.estimated_depth || 'N/A'}</td>
                </tr>`;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>`;
        
        // Add pagination if needed
        if (totalPages > 1) {
            html += `
                <div class="table-pagination">
                    <span>Showing ${Math.min(PAGE_SIZE, cratersData.length)} of ${cratersData.length} craters</span>
                    <div class="pagination-controls">
                        <button class="btn small-btn" id="prev-page" disabled>Previous</button>
                        <span id="page-indicator">Page 1 of ${totalPages}</span>
                        <button class="btn small-btn" id="next-page">Next</button>
                    </div>
                </div>
                <input type="hidden" id="current-page" value="1">
                <input type="hidden" id="total-pages" value="${totalPages}">`;
        }
        
        html += `</div>`;
    } else {
        html += `<div class="table-section">
                    <h3>Detected Craters</h3>
                    <div class="empty-table-message">No crater data available</div>
                </div>`;
    }
    
    html += '</div>';
    return html;
}

// Setup table pagination
function setupTablePagination(container, jsonData) {
    if (!container) return;
    
    // Determine the craters data source
    const cratersData = jsonData.craters || (Array.isArray(jsonData) ? jsonData : []);
    
    // If no pagination needed, exit early
    if (cratersData.length <= 10) return;
    
    const PAGE_SIZE = 10;
    const totalPagesEl = container.querySelector('#total-pages');
    if (!totalPagesEl) return;
    
    const totalPages = parseInt(totalPagesEl.value);
    const prevBtn = container.querySelector('#prev-page');
    const nextBtn = container.querySelector('#next-page');
    const pageIndicator = container.querySelector('#page-indicator');
    const currentPageInput = container.querySelector('#current-page');
    
    if (!prevBtn || !nextBtn || !pageIndicator || !currentPageInput) return;
    
    // Setup event listeners for pagination buttons
    prevBtn.addEventListener('click', function() {
        let currentPage = parseInt(currentPageInput.value);
        if (currentPage > 1) {
            currentPage--;
            updateCratersTable(currentPage);
        }
    });
    
    nextBtn.addEventListener('click', function() {
        let currentPage = parseInt(currentPageInput.value);
        if (currentPage < totalPages) {
            currentPage++;
            updateCratersTable(currentPage);
        }
    });
    
    // Function to update the table content based on current page
    function updateCratersTable(page) {
        // Update page controls
        currentPageInput.value = page;
        pageIndicator.textContent = `Page ${page} of ${totalPages}`;
        prevBtn.disabled = page === 1;
        nextBtn.disabled = page === totalPages;
        
        // Update table content
        const start = (page - 1) * PAGE_SIZE;
        const end = Math.min(start + PAGE_SIZE, cratersData.length);
        const cratesToShow = cratersData.slice(start, end);
        
        const tableBody = container.querySelector('#craters-table tbody');
        if (!tableBody) return;
        
        tableBody.innerHTML = '';
        
        cratesToShow.forEach(crater => {
            const row = document.createElement('tr');
            
            // ID and confidence
            row.innerHTML = `
                <td>${crater.id || 'N/A'}</td>
                <td>${crater.confidence || 0}%</td>
                <td>${crater.bounding_box?.width || 0} × ${crater.bounding_box?.height || 0}</td>
                <td>Center: (${crater.center?.x || 0}, ${crater.center?.y || 0})</td>
                <td>${crater.estimated_depth || 'N/A'}</td>`;
            
            tableBody.appendChild(row);
        });
    }
}

// Update technical data with detection results
function updateTechnicalData(data) {
    console.log("Updating technical data with:", data);
    
    // Store the data globally for use in buttons
    window.craterData = null;
    
    // Show the technical data section
    const technicalSection = document.getElementById('technical-data');
    if (technicalSection) {
        technicalSection.classList.remove('hidden');
    } else {
        console.error("Technical data section not found");
        return;
    }
    
    // Process image data only
    let craters = [];
    let totalWidth = 0;
    let totalHeight = 0;
    let totalConfidence = 0;
    
    // Process image data
    if (data.detections && Array.isArray(data.detections)) {
        craters = data.detections.map((detection, index) => {
            // Extract bounding box data
            const [x1, y1, x2, y2] = detection.bbox;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Calculate center
            const centerX = x1 + width / 2;
            const centerY = y1 + height / 2;
            
            // Track totals for averages
            totalWidth += width;
            totalHeight += height;
            totalConfidence += detection.confidence;
            
            // Return formatted crater data
            return {
                id: index + 1,
                confidence: Math.round(detection.confidence * 10000) / 100,
                class: detection.class_id || 0,
                bounding_box: {
                    x1: Math.round(x1),
                    y1: Math.round(y1),
                    x2: Math.round(x2),
                    y2: Math.round(y2),
                    width: Math.round(width),
                    height: Math.round(height)
                },
                center: {
                    x: Math.round(centerX),
                    y: Math.round(centerY)
                },
                estimated_depth: Math.round((width + height) / 20) + " units"
            };
        });
    }
    
    // Calculate averages
    const craterCount = craters.length;
    const avgWidth = craterCount > 0 ? Math.round(totalWidth / craterCount) : 0;
    const avgHeight = craterCount > 0 ? Math.round(totalHeight / craterCount) : 0;
    const avgConfidence = craterCount > 0 ? Math.round((totalConfidence / craterCount) * 10000) / 100 : 0;
    
    // Create final data object (image-only structure)
    const craterData = {
        summary: {
            total_craters: craterCount,
            average_size: {
                width: avgWidth,
                height: avgHeight
            },
            average_confidence: avgConfidence,
            detection_time: data.detection_time || new Date().toISOString(),
            media_type: "image"
        },
        media_info: {
            name: data.image_name || "Unknown",
            type: "image",
            dimensions: data.image_dimensions || { width: 0, height: 0 },
            path: data.result_path || "Unknown"
        },
        craters: craters
    };
    
    // Store data globally for other functions
    window.craterData = craterData;
    
    // Update the display using table format by default
    const jsonContainer = document.getElementById('crater-data-json');
    if (jsonContainer) {
        // Set the appropriate class for table mode
        jsonContainer.classList.add('table-mode');
        jsonContainer.classList.remove('json-mode');
        
        // Generate and display table HTML
        const tableHTML = convertJSONToTable(craterData);
        jsonContainer.innerHTML = tableHTML;
        
        // Setup pagination if needed
        setupTablePagination(jsonContainer, craterData);
        
        // Update toggle button text to reflect current state
        const toggleFormatBtn = document.getElementById('toggle-format-btn');
        if (toggleFormatBtn) {
            toggleFormatBtn.textContent = 'Switch to JSON View';
        }
    } else {
        console.error("JSON container element not found");
    }
    
    // Update statistics
    const craterCountEl = document.getElementById('crater-count');
    const avgSizeEl = document.getElementById('avg-size');
    const avgConfidenceEl = document.getElementById('avg-confidence');
    
    if (craterCountEl) craterCountEl.textContent = craterData.summary.total_craters;
    if (avgSizeEl) {
        const avgSize = Math.round((avgWidth + avgHeight) / 2);
        avgSizeEl.textContent = avgSize + ' px';
    }
    if (avgConfidenceEl) avgConfidenceEl.textContent = craterData.summary.average_confidence + '%';
}

// Initialize 3D visualization
function initialize3DVisualization(data) {
    console.log("Initializing 3D visualization...");
    
    const visualizationSection = document.getElementById('visualization-3d');
    
    if (!visualizationSection) {
        console.error('3D visualization section not found');
        return;
    }
    
    // Show the visualization section
    visualizationSection.classList.remove('hidden');
    
    try {
        // Initialize the 3D visualizer if not already done
        if (window.craterVisualizer && !window.craterVisualizer.initialized) {
            window.craterVisualizer.init();
        }
        
        // If initialization failed, return early
        if (!window.craterVisualizer || !window.craterVisualizer.initialized) {
            console.error("3D visualization initialization failed");
            return;
        }
        
        // For image data
        if (data.detections) {
            window.craterVisualizer.createTerrain(
                data.result_path, 
                data.detections
            );
        } else {
            console.warn("No detection data available for 3D visualization");
        }
    } catch (error) {
        console.error("Error initializing 3D visualization:", error);
    }
}