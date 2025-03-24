// DOM Elements
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const resultsSection = document.getElementById('results-section');
const loadingSpinner = document.getElementById('loading-spinner');
const resultsContainer = document.getElementById('results-container');
const originalImage = document.getElementById('original-image');
const detectedImage = document.getElementById('detected-image');
const downloadBtn = document.getElementById('download-btn');
const newAnalysisBtn = document.getElementById('new-analysis-btn');

// Event Listeners
dropArea.addEventListener('dragover', handleDragOver);
dropArea.addEventListener('dragleave', handleDragLeave);
dropArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
downloadBtn.addEventListener('click', handleDownload);
newAnalysisBtn.addEventListener('click', resetAnalysis);

// Functions

//Reload
window.onload = function () {
    document.getElementById("hero").scrollIntoView({ behavior: "smooth" });
};



//Scroll Down
document.addEventListener('DOMContentLoaded', function() {
    // Get the explore button element
    const exploreBtn = document.getElementById('exploreBtn');
    
    // Add click event listener to the button
    if (exploreBtn) {
        exploreBtn.addEventListener('click', function() {
            // Scroll to the steps section smoothly
            const stepsSection = document.querySelector('.steps-section');
            if (stepsSection) {
                stepsSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
});

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropArea.classList.add('highlight');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropArea.classList.remove('highlight');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropArea.classList.remove('highlight');
    
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Check file type
    if (!['image/jpeg', 'image/png', 'image/gif'].includes(file.type)) {
        alert('Please upload a JPG or PNG image');
        return;
    }
    
    // Check file size (2MB max)
    if (file.size > 2 * 1024 * 1024) {
        alert('File size should be less than 2MB');
        return;
    }
    
    // Display the original image
    const reader = new FileReader();
    reader.onload = function(e) {
        originalImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // Show the results section and loading spinner
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    loadingSpinner.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    // Send to API
    fetch('/detect/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.detail || 'Error detecting craters');
            });
        }
        return response.blob();
    })
    .then(blob => {
        const imageUrl = URL.createObjectURL(blob);
        
        // Load the image first
        const img = new Image();
        img.onload = function() {
            // Once image is loaded, hide spinner and show results
            loadingSpinner.classList.add('hidden');
            resultsContainer.classList.remove('hidden');
            
            // Add animation classes
            resultsContainer.classList.add('fade-in');
            
            // Set the image source
            detectedImage.src = imageUrl;
            
            // Start the simplified scanning animation
            startScanningAnimation();
        };
        img.src = imageUrl;
        
        // Store the blob URL for download
        downloadBtn.setAttribute('data-url', imageUrl);
    })
    .catch(error => {
        alert('Error: ' + error.message);
        loadingSpinner.classList.add('hidden');
    });
}

function startScanningAnimation() {
    // Create scanning overlay
    const detectedImageContainer = document.querySelector('.detected-image');
    const scanningOverlay = document.createElement('div');
    scanningOverlay.className = 'scanning-overlay';
    
    // Add scanning text
    const scanningText = document.createElement('div');
    scanningText.className = 'scanning-text';
    scanningText.textContent = 'Analyzing surface...';
    scanningOverlay.appendChild(scanningText);
    
    // Add scan line
    const scanLine = document.createElement('div');
    scanLine.className = 'scan-line';
    scanningOverlay.appendChild(scanLine);
    
    // Add overlay to container
    detectedImageContainer.appendChild(scanningOverlay);
    
    // Create multiple scan lines for effect
    setTimeout(() => {
        const scanLine2 = document.createElement('div');
        scanLine2.className = 'scan-line';
        scanLine2.style.animationDelay = '0.7s';
        scanningOverlay.appendChild(scanLine2);
    }, 500);
    
    // Final update
    setTimeout(() => {
        scanningText.textContent = 'Analysis complete';
    }, 2000);
    
    // Remove scanning overlay after animation completes
    setTimeout(() => {
        scanningOverlay.style.opacity = '0';
        scanningOverlay.style.transition = 'opacity 0.5s ease';
        
        setTimeout(() => {
            detectedImageContainer.removeChild(scanningOverlay);
            detectedImage.classList.add('scan-complete');
        }, 500);
    }, 3000);
}

function handleDownload() {
    const imageUrl = downloadBtn.getAttribute('data-url');
    if (!imageUrl) return;
    
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = 'crater-detection-result.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Add animation to show success
    downloadBtn.classList.add('pulse');
    setTimeout(() => {
        downloadBtn.classList.remove('pulse');
    }, 1000);
}

function resetAnalysis() {
    resultsSection.classList.add('hidden');
    fileInput.value = '';
    originalImage.src = '';
    detectedImage.src = '';
    resultsContainer.classList.remove('fade-in');
    
    // Scroll back to upload section
    document.querySelector('.upload-section').scrollIntoView({ behavior: 'smooth' });
}

// Add animation to the upload icon
const uploadIcon = document.querySelector('.upload-icon i');
setInterval(() => {
    uploadIcon.classList.add('bounce');
    setTimeout(() => {
        uploadIcon.classList.remove('bounce');
    }, 1000);
}, 3000);

// Initialize tooltips for better user experience
const buttons = document.querySelectorAll('.btn');
buttons.forEach(button => {
    button.setAttribute('title', button.textContent.trim());
});

// Add initial animations
window.addEventListener('load', () => {
    document.querySelector('.upload-container').classList.add('slide-in');
});