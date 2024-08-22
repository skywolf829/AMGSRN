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

    function generateRandom3DTexture(size) {
        const data = new Float32Array(size * size * size);
        for (let i = 0; i < size * size * size; i++) {
            data[i] = Math.random();
        }
    
        const texture = new THREE.Data3DTexture(data, size, size, size);
        texture.format = THREE.RedFormat;
        texture.type = THREE.FloatType;
        texture.minFilter = THREE.LinearFilter;
        texture.magFilter = THREE.LinearFilter;
        texture.unpackAlignment = 1;
        return texture;
    }

    async function setupShaderMaterial(jsonData, binDataDict) {
       
    }

});