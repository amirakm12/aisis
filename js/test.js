// Simple test to verify Three.js is loading correctly
import * as THREE from 'three';

console.log('Three.js version:', THREE.REVISION);
console.log('WebGL supported:', !!window.WebGLRenderingContext);

// Test basic Three.js functionality
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();

console.log('Three.js basic objects created successfully');
console.log('Scene:', scene);
console.log('Camera:', camera);
console.log('Renderer:', renderer);