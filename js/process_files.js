import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

document.addEventListener('DOMContentLoaded', function() {

    const dropArea = document.getElementById('drop-area');
    const jsonContent = document.getElementById('json-content');
    const binContent = document.getElementById('bin-content');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop area when item is dragged over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
    });

    // Handle the drop event
    dropArea.addEventListener('drop', handleDrop, false);

    setupShaderMaterial(null, null);

    async function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length !== 1 || !files[0].name.endsWith('.zip')) {
            alert("Please drop exactly one .zip file.");
            return;
        }

        const zipFile = files[0];
        const zip = new JSZip();

        try {
            const content = await zip.loadAsync(zipFile);
            let jsonData = null;
            const binDataDict = {};

            const jsonFile = content.file("options.json");
            if (jsonFile) {
                const jsonText = await jsonFile.async("string");
                jsonData = JSON.parse(jsonText);
                document.getElementById('json-content').textContent = JSON.stringify(jsonData, null, 2);
            }

            const binFiles = Object.keys(content.files).filter(name => name.endsWith('.bin'));
            for (const binFileName of binFiles) {
                const binFile = content.file(binFileName);
                const binArrayBuffer = await binFile.async("arraybuffer");
                const floatArray = new Float32Array(binArrayBuffer);
                binDataDict[binFileName] = floatArray;

                document.getElementById('bin-content').textContent += `${binFileName}: ${floatArray.length} floats\n`;
            }

            setupShaderMaterial(jsonData, binDataDict);

        } catch (err) {
            console.error("Error reading ZIP file:", err);
        }
    }

    async function loadShaderFile(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load shader: ${url}`);
        }
        return await response.text();
    }

    async function setupShaderMaterial(jsonData, binDataDict) {
        try {
            const vertexShader = await loadShaderFile('./shaders/vertex.glsl');
            const fragmentShader = await loadShaderFile('./shaders/fragment.glsl');
    
            const material = new THREE.ShaderMaterial({
                vertexShader: vertexShader,
                fragmentShader: fragmentShader,
                uniforms: {
                    //uFloatArray: { value: binDataDict['somefile.bin'] }
                }
            });
    
            const geometry = new THREE.BoxGeometry(2, 2, 2);
            const mesh = new THREE.Mesh(geometry, material);
    
            const scene = new THREE.Scene();
            scene.add(mesh);
    
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth * 0.75 / window.innerHeight, 0.1, 1000);
            camera.position.z = 5;
    
            const renderer = new THREE.WebGLRenderer();
            renderer.setSize(window.innerWidth * 0.75, window.innerHeight);
            document.getElementById('renderer-container').appendChild(renderer.domElement);
    
            // Initialize OrbitControls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true; // Optional: adds damping (inertia) to the controls
            controls.dampingFactor = 0.05;
            controls.screenSpacePanning = false;
            controls.maxPolarAngle = Math.PI / 2;

            function animate() {
                requestAnimationFrame(animate);
                controls.update(); // Only required if controls.enableDamping = true, or if controls.autoRotate = true
                renderer.render(scene, camera);
            }
            animate();
        } catch (error) {
            console.error('Error setting up shader material:', error);
        }
    }

});