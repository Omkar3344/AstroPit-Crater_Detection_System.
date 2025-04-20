/**
 * Crater 3D Visualization with Three.js
 * This file handles the 3D visualization of detected craters.
 */

// Main visualization class
class CraterVisualizer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.terrain = null;
        this.craters = [];
        this.wireframeMode = false;
        this.depthFactor = 0.3; // Default depth factor for craters
        this.sizeFactor = 1; // Default size factor for terrain
        this.container = document.getElementById('crater-3d-container');
        this.initialized = false;
        this.imageTexture = null;
        this.heightMap = null;
        this.loadingManager = new THREE.LoadingManager();
        this.orbitAnimation = null;
        this.originalCameraPosition = null;
        
        // Set loading manager events
        this.loadingManager.onStart = () => this.showLoading(true);
        this.loadingManager.onLoad = () => this.showLoading(false);
        this.loadingManager.onError = (url) => {
            console.error('Error loading 3D resource:', url);
            this.showLoading(false);
        };
    }
    
    // Initialize the 3D scene
    init() {
        if (this.initialized) return;
        
        console.log("Initializing 3D visualization...");
        
        // Check if Three.js is available
        if (!window.THREE) {
            console.error("Three.js is not loaded! Cannot initialize 3D visualization.");
            this.showError("Three.js library is not available");
            return;
        }
        
        // Check if container exists
        if (!this.container) {
            console.error("3D container element not found!");
            return;
        }
        
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x050a1f);
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            60, // Field of view
            this.container.clientWidth / this.container.clientHeight, // Aspect ratio
            0.1, // Near clipping plane
            1000 // Far clipping plane
        );
        this.camera.position.set(0, 10, 10);
        this.camera.lookAt(0, 0, 0);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.container.appendChild(this.renderer.domElement);
        
        // Add controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Replace the lighting setup in the init() method
        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6); // Brighter ambient
        this.scene.add(ambientLight);

        // Add directional light (sun)
        const sunLight = new THREE.DirectionalLight(0xffffff, 0.8); // Natural white light
        sunLight.position.set(3, 10, 5);
        sunLight.castShadow = true;
        this.scene.add(sunLight);

        // Add fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, 3, -5);
        this.scene.add(fillLight);
        
        // Add stars to background
        this.addStars();
        
        // Add visual effects
        this.addVisualEffects();
        
        // Set up resize handler
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Start animation loop
        this.animate();
        
        this.initialized = true;
    }
    
    // Add stars to the background
    addStars() {
        const starsGeometry = new THREE.BufferGeometry();
        const starsMaterial = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.1,
            transparent: true,
            opacity: 0.8
        });
        
        const starsVertices = [];
        for (let i = 0; i < 1000; i++) {
            const x = (Math.random() - 0.5) * 100;
            const y = (Math.random() - 0.5) * 100;
            const z = (Math.random() - 0.5) * 100;
            starsVertices.push(x, y, z);
        }
        
        starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starsVertices, 3));
        const stars = new THREE.Points(starsGeometry, starsMaterial);
        this.scene.add(stars);
    }
    
    // Either completely remove the addVisualEffects method or replace it with this minimal version
    addVisualEffects() {
        // Add only very subtle dust particles for scale reference
        const particleCount = 500;
        const particles = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount * 3; i += 3) {
            const radius = 20 + Math.random() * 10;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.random() * Math.PI - Math.PI/2;
            
            positions[i] = radius * Math.cos(theta) * Math.cos(phi);
            positions[i+1] = radius * Math.sin(phi);
            positions[i+2] = radius * Math.sin(theta) * Math.cos(phi);
        }
        
        particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const particleMaterial = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.02,
            transparent: true,
            opacity: 0.2
        });
        
        const particleSystem = new THREE.Points(particles, particleMaterial);
        this.scene.add(particleSystem);
    }
    
    // Create terrain from image
    createTerrain(imageUrl, detections) {
        // Show loading indicator
        this.showLoading(true);
        
        // Remove previous terrain if exists
        if (this.terrain) {
            this.scene.remove(this.terrain);
            this.terrain.geometry.dispose();
            this.terrain.material.dispose();
        }
        
        // Clear previous craters
        this.clearCraters();
        
        // Load image texture
        const textureLoader = new THREE.TextureLoader(this.loadingManager);
        textureLoader.load(imageUrl, (texture) => {
            this.imageTexture = texture;
            
            // Create plane geometry with thickness
            const aspectRatio = texture.image.width / texture.image.height;
            let planeWidth = 10;
            let planeHeight = planeWidth / aspectRatio;

            // Create a box instead of a plane to add thickness
            const thickness = 0.5; // Add thickness to make it look like land
            const geometry = new THREE.BoxGeometry(planeWidth, thickness, planeHeight, 128, 1, 128);
            const positionAttribute = geometry.attributes.position;
            const vertices = positionAttribute.array;

            // Add irregular surface to the top face only (y > 0)
            for (let i = 0; i < vertices.length; i += 3) {
                // Check if this is a top vertex (y is close to thickness/2)
                if (Math.abs(vertices[i + 1] - thickness / 2) < 0.01) {
                    // Add random height variation - smaller values keep subtlety
                    const noise = (Math.random() * 0.04) - 0.02;
                    vertices[i + 1] += noise;
                }
            }

            // Update the geometry
            positionAttribute.needsUpdate = true;

            // Create materials for the terrain with a more realistic, non-shiny appearance
            const topMaterial = new THREE.MeshStandardMaterial({
                map: texture,
                roughness: 0.95,  // Increase roughness significantly to eliminate shininess
                metalness: 0.0,   // Remove all metalness - planetary surfaces are not metallic
                flatShading: true, // Enable flat shading for a more rugged appearance
                displacementScale: 0.05,
                displacementBias: -0.05
            });

            // Create materials for sides and bottom (slightly varied colors for realism)
            const sideMaterial = new THREE.MeshStandardMaterial({
                color: 0x807060, // Slightly different earthy tone
                roughness: 1.0,   // Completely rough
                metalness: 0.0,   // No metalness
                flatShading: true // Flat shading for rocky appearance
            });

            const bottomMaterial = new THREE.MeshStandardMaterial({
                color: 0x605040, // Darker tone for bottom
                roughness: 1.0,
                metalness: 0.0,
                flatShading: true
            });

            // Create an array of materials for each face of the box with slight variations
            const materials = [
                sideMaterial,    // right side
                sideMaterial,    // left side
                topMaterial,     // top (with image texture)
                bottomMaterial,  // bottom
                new THREE.MeshStandardMaterial({
                    color: 0x756b5a, // Slight variation for front
                    roughness: 0.9,
                    metalness: 0.0,
                    flatShading: true
                }),
                new THREE.MeshStandardMaterial({
                    color: 0x6c6355, // Slight variation for back
                    roughness: 0.9,
                    metalness: 0.0,
                    flatShading: true
                })
            ];

            // Create terrain mesh with multiple materials
            this.terrain = new THREE.Mesh(geometry, materials);

            // Position the terrain so the top face is at y=0
            this.terrain.position.y = -thickness / 2;

            // No need to rotate as we're using a box with the texture on top
            this.terrain.receiveShadow = true;
            this.scene.add(this.terrain);
            
            // Add a glowing edge to outline the terrain
            const edgeGeometry = new THREE.EdgesGeometry(geometry, 15); // 15 degree threshold
            const edgeMaterial = new THREE.LineBasicMaterial({ 
                color: 0x444444,
                transparent: true,
                opacity: 0.5
            });
            const wireframe = new THREE.LineSegments(edgeGeometry, edgeMaterial);
            this.terrain.add(wireframe);
            
            // Add bounding boxes for craters
            if (detections && detections.length > 0) {
                this.addCraters(detections, texture.image.width, texture.image.height, planeWidth, planeHeight);
                document.querySelector('.crater-count-3d').textContent = detections.length;
            }
            
            // Reset camera view
            this.resetCamera();
            
            // Hide loading indicator
            this.showLoading(false);
        }, undefined, (error) => {
            console.error('Error loading texture:', error);
            this.showLoading(false);
        });
    }
    
    // Add craters based on detection data
    addCraters(detections, imgWidth, imgHeight, planeWidth, planeHeight) {
        // Ensure detections is a valid array
        if (!detections || !Array.isArray(detections) || detections.length === 0) {
            console.warn("No valid detections provided for 3D visualization");
            return;
        }
        
        // Loop through all crater detections
        detections.forEach(detection => {
            // Skip invalid detections
            if (!detection || !detection.bbox) {
                console.warn("Skipping invalid detection without bbox");
                return; // Continue to next detection
            }
            
            // Extract bounding box coordinates
            const [x1, y1, x2, y2] = detection.bbox;
            
            // Calculate dimensions
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Calculate center position in image coordinates
            const centerX = x1 + width / 2;
            const centerY = y1 + height / 2;
            
            // Convert image coordinates to 3D space coordinates
            const normX = (centerX / imgWidth) - 0.5;
            const normY = (centerY / imgHeight) - 0.5;
            
            // Scale to plane dimensions and adjust for plane rotation
            const worldX = normX * planeWidth;
            const worldZ = -normY * planeHeight; // Negative because of plane rotation
            
            // Scale width and height to 3D space
            const craterWidth = (width / imgWidth) * planeWidth;
            const craterHeight = (height / imgHeight) * planeHeight;
            const craterSize = Math.max(craterWidth, craterHeight);
            
            // Create crater visualization
            const craterRadius = craterSize / 2;
            
            // Create depth based on size and confidence
            const depth = craterSize * (0.2 + (detection.confidence * 0.8)) * this.depthFactor;
            
            // Create crater with different visualizations
            this.createCraterGeometry(worldX, worldZ, craterRadius, depth, detection.confidence);
        });
    }
    
    // Replace the current createCraterGeometry function with this version that doesn't add visible markers
    createCraterGeometry(x, z, radius, depth, confidence) {
        // Don't create any visible geometry for craters
        // This function is intentionally empty to prevent adding bounding boxes
        // We'll still count the craters for statistics, but won't render them

        // If you want to maintain the structure without visible elements, 
        // you can add a negligible, invisible placeholder to preserve the array structure
        const dummyGeometry = new THREE.BufferGeometry();
        const dummyMaterial = new THREE.Material();
        dummyMaterial.visible = false;
        
        const dummy = new THREE.Mesh(dummyGeometry, dummyMaterial);
        dummy.visible = false;
        this.scene.add(dummy);
        this.craters.push(dummy);
        
        // This ensures crater counting works while nothing is actually visible
    }
    
    // Clear all crater visualizations
    clearCraters() {
        this.craters.forEach(crater => {
            this.scene.remove(crater);
            if (crater.geometry) crater.geometry.dispose();
            if (crater.material) crater.material.dispose();
        });
        this.craters = [];
    }
    
    // Toggle wireframe mode for all materials
    toggleWireframe() {
        this.wireframeMode = !this.wireframeMode;
        
        // Update terrain material
        if (this.terrain) {
            this.terrain.material.wireframe = this.wireframeMode;
        }
        
        // Update crater materials
        this.craters.forEach(crater => {
            if (crater.material) {
                crater.material.wireframe = this.wireframeMode;
            }
        });
    }
    
    // Update setDepthFactor to work with the new terrain structure
    setDepthFactor(factor) {
        this.depthFactor = factor;
        
        // Simply recreate the craters without changing the terrain
        if (this.imageTexture && window.craterData) {
            // Get detections
            let detections;
            
            // Handle different data structures
            if (Array.isArray(window.craterData)) {
                detections = window.craterData;
            } else if (window.craterData.craters && Array.isArray(window.craterData.craters)) {
                detections = window.craterData.craters;
            } else {
                console.warn("No valid crater data found for 3D visualization");
                return;
            }
            
            // Check if we have detections
            if (!detections || detections.length === 0) {
                console.warn("No detections available for 3D visualization");
                return;
            }
            
            const imgWidth = this.imageTexture.image.width;
            const imgHeight = this.imageTexture.image.height;
            
            // Clear existing craters
            this.clearCraters();
            
            // Recalculate the plane size
            const aspectRatio = imgWidth / imgHeight;
            let planeWidth = 10;
            let planeHeight = planeWidth / aspectRatio;
            
            // Add craters with new depth 
            this.addCraters(detections, imgWidth, imgHeight, planeWidth, planeHeight);
        }
    }
    
    // Add this method to the CraterVisualizer class
    setSizeFactor(factor) {
        // Store the size factor
        this.sizeFactor = factor;
        
        // Update terrain scale
        if (this.terrain) {
            this.terrain.scale.set(1, factor, 1);
        }
    }
    
    // Reset camera to default position
    resetCamera() {
        this.camera.position.set(0, 5, 8);  // Move camera closer
        this.camera.lookAt(0, 0, 0);
        this.controls.reset();
    }
    
    // Handle window resize
    onWindowResize() {
        if (!this.camera || !this.renderer || !this.container) return;
        
        // Update camera aspect ratio
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        
        // Update renderer size
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    // Animation loop
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update controls
        if (this.controls) {
            this.controls.update();
        }
        
        // Render the scene
        if (this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    // Show or hide loading indicator
    showLoading(show) {
        const loadingElement = document.getElementById('loading-3d');
        if (loadingElement) {
            if (show) {
                loadingElement.classList.add('active');
            } else {
                loadingElement.classList.remove('active');
            }
        }
    }
    
    // Show error message
    showError(message) {
        const loadingElement = document.getElementById('loading-3d');
        if (loadingElement) {
            loadingElement.classList.add('active');
            loadingElement.innerHTML = `
                <div class="error-icon"><i class="fas fa-exclamation-triangle"></i></div>
                <span>Error: ${message}</span>
            `;
        }
    }

    // Update the startOrbitalFlight method to support zooming
    startOrbitalFlight() {
        if (this.orbitAnimation) {
            // Stop existing animation
            cancelAnimationFrame(this.orbitAnimation);
            this.orbitAnimation = null;
            
            // Re-enable all controls features
            this.controls.enableRotate = true;
            this.controls.enablePan = true;
            
            return false; // Animation stopped
        }
        
        // Store original camera position to restore later
        this.originalCameraPosition = {
            x: this.camera.position.x,
            y: this.camera.position.y,
            z: this.camera.position.z
        };
        
        // Setup animation parameters
        let time = 0;
        const duration = 30; // seconds
        const orbitSpeed = 0.5;
        const baseOrbitRadius = 12;
        const baseOrbitHeight = 6;
        
        // Disable rotation and panning, but keep zoom
        this.controls.enableRotate = false;
        this.controls.enablePan = false;
        this.controls.enableZoom = true;
        
        // Animation function with zoom support
        const animate = () => {
            time += 0.01;
            
            // Get current distance from center (for zoom)
            const currentDistance = this.camera.position.length();
            const defaultDistance = Math.sqrt(baseOrbitRadius*baseOrbitRadius + baseOrbitHeight*baseOrbitHeight);
            const zoomRatio = currentDistance / defaultDistance;
            
            // Calculate camera position in orbit with current zoom level
            const angle = time * orbitSpeed;
            const effectiveRadius = baseOrbitRadius * zoomRatio;
            const effectiveHeight = baseOrbitHeight * zoomRatio;
            
            const x = effectiveRadius * Math.cos(angle);
            const z = effectiveRadius * Math.sin(angle);
            
            // Move camera in orbit
            this.camera.position.set(x, effectiveHeight, z);
            this.camera.lookAt(0, 0, 0);
            
            if (time < duration) {
                this.orbitAnimation = requestAnimationFrame(animate);
            } else {
                // Return to original position when done
                cancelAnimationFrame(this.orbitAnimation);
                this.orbitAnimation = null;
                
                // Re-enable control features
                this.controls.enableRotate = true;
                this.controls.enablePan = true;
                
                // Smooth transition back to original position
                this.transitionCamera();
            }
        };
        
        // Start animation
        this.orbitAnimation = requestAnimationFrame(animate);
        return true; // Animation started
    }

    // Add smooth camera transition method
    transitionCamera(callback) {
        if (!this.originalCameraPosition) return;
        
        const startPosition = {
            x: this.camera.position.x,
            y: this.camera.position.y,
            z: this.camera.position.z
        };
        
        const endPosition = this.originalCameraPosition;
        let progress = 0;
        const duration = 2; // seconds
        
        const transition = () => {
            progress += 0.01;
            const t = Math.min(progress / duration, 1);
            
            // Use smoothstep for easing
            const smoothT = t * t * (3 - 2 * t);
            
            // Interpolate position
            this.camera.position.x = startPosition.x + (endPosition.x - startPosition.x) * smoothT;
            this.camera.position.y = startPosition.y + (endPosition.y - startPosition.y) * smoothT;
            this.camera.position.z = startPosition.z + (endPosition.z - startPosition.z) * smoothT;
            
            this.camera.lookAt(0, 0, 0);
            
            if (t < 1) {
                requestAnimationFrame(transition);
            } else if (callback) {
                callback();
            }
        };
        
        requestAnimationFrame(transition);
    }
}

// Create global instance of the visualizer
window.craterVisualizer = new CraterVisualizer();

// Initialize event listeners when document is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize control buttons
    const toggleWireframeBtn = document.getElementById('toggle-wireframe');
    const toggleElevationBtn = document.getElementById('toggle-elevation');
    const craterDepthSlider = document.getElementById('crater-depth');
    const resetCameraBtn = document.getElementById('reset-camera');
    
    // Wireframe toggle
    if (toggleWireframeBtn) {
        toggleWireframeBtn.addEventListener('click', function() {
            if (window.craterVisualizer) {
                window.craterVisualizer.toggleWireframe();
            }
        });
    }
    
    // Remove or modify the craterDepthSlider event listener
    if (craterDepthSlider) {
        // Either remove this entirely
        craterDepthSlider.parentElement.style.display = 'none';
        
        // Or repurpose it for something else like crater size multiplier
        craterDepthSlider.addEventListener('input', function() {
            if (window.craterVisualizer) {
                const sizeFactor = parseFloat(this.value) / 50;
                window.craterVisualizer.setSizeFactor(sizeFactor);
            }
        });
    }
    
    // Reset camera
    if (resetCameraBtn) {
        resetCameraBtn.addEventListener('click', function() {
            if (window.craterVisualizer) {
                window.craterVisualizer.resetCamera();
            }
        });
    }
    
    // Remove the toggleElevation button or repurpose it
    if (toggleElevationBtn) {
        toggleElevationBtn.style.display = 'none';
    }

    // Add to the DOMContentLoaded event listener
    const orbitalFlightBtn = document.getElementById('orbital-flight');
    if (orbitalFlightBtn) {
        orbitalFlightBtn.addEventListener('click', function() {
            if (window.craterVisualizer) {
                const isStarting = window.craterVisualizer.startOrbitalFlight();
                this.textContent = isStarting ? 'Stop Orbit' : 'Orbital Flight';
            }
        });
    }
});

// Update the 3D visualization function to handle video data more effectively
function initialize3DVisualization(data) {
    console.log("Initializing 3D visualization...");
    
    const visualizationSection = document.getElementById('visualization-3d');
    
    if (!visualizationSection) {
        console.error('3D visualization section not found');
        return;
    }
    
    // Show the visualization section
    visualizationSection.classList.remove('hidden');
    
    // Return early if no crater data is available
    if (!data || !data.detections) {
        console.warn("No crater detection data available for 3D visualization");
        if (window.craterVisualizer) {
            window.craterVisualizer.showError("No crater detection data available");
        }
        return;
    }
    
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
        window.craterVisualizer.createTerrain(
            data.result_path, 
            data.detections
        );
    } catch (error) {
        console.error("Error initializing 3D visualization:", error);
    }
}
