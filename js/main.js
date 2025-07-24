import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

class GreekGoddessAvatar {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.composer = null;
        this.avatar = null;
        this.animationMixer = null;
        this.clock = new THREE.Clock();
        
        // Animation states
        this.isAnimating = false;
        this.currentExpression = 'neutral';
        this.lightingMode = 'divine';
        
        this.init();
    }
    
    init() {
        this.createScene();
        this.createCamera();
        this.createRenderer();
        this.createControls();
        this.createLighting();
        this.createAvatar();
        this.createPostProcessing();
        this.setupEventListeners();
        this.animate();
        
        // Hide loading screen
        setTimeout(() => {
            document.getElementById('loading').style.display = 'none';
        }, 2000);
    }
    
    createScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        
        // Add fog for atmospheric effect
        this.scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);
    }
    
    createCamera() {
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(0, 2, 8);
    }
    
    createRenderer() {
        const canvas = document.getElementById('avatar-canvas');
        this.renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
    }
    
    createControls() {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 3;
        this.controls.maxDistance = 15;
        this.controls.maxPolarAngle = Math.PI / 1.8;
    }
    
    createLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scene.add(ambientLight);
        
        // Main key light (divine glow)
        const keyLight = new THREE.DirectionalLight(0x00ffff, 1.5);
        keyLight.position.set(5, 10, 5);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        this.scene.add(keyLight);
        
        // Rim light
        const rimLight = new THREE.DirectionalLight(0x8a2be2, 0.8);
        rimLight.position.set(-5, 3, -5);
        this.scene.add(rimLight);
        
        // Fill light
        const fillLight = new THREE.PointLight(0xffffff, 0.5, 10);
        fillLight.position.set(0, 5, 3);
        this.scene.add(fillLight);
        
        // Ethereal particles
        this.createParticles();
    }
    
    createParticles() {
        const particleCount = 1000;
        const particles = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 20;
            positions[i + 1] = (Math.random() - 0.5) * 20;
            positions[i + 2] = (Math.random() - 0.5) * 20;
            
            colors[i] = Math.random() * 0.5 + 0.5;     // R
            colors[i + 1] = Math.random() * 0.5 + 0.5; // G
            colors[i + 2] = 1;                         // B
        }
        
        particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        particles.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const particleMaterial = new THREE.PointsMaterial({
            size: 0.1,
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending
        });
        
        const particleSystem = new THREE.Points(particles, particleMaterial);
        this.scene.add(particleSystem);
        
        // Animate particles
        this.animateParticles = () => {
            const positions = particleSystem.geometry.attributes.position.array;
            for (let i = 0; i < positions.length; i += 3) {
                positions[i + 1] += Math.sin(Date.now() * 0.001 + i) * 0.01;
            }
            particleSystem.geometry.attributes.position.needsUpdate = true;
            particleSystem.rotation.y += 0.002;
        };
    }
    
    createAvatar() {
        this.avatar = new THREE.Group();
        
        // Create head
        this.createHead();
        
        // Create hair
        this.createHair();
        
        // Create eyes
        this.createEyes();
        
        // Create accessories (Greek goddess elements)
        this.createAccessories();
        
        // Create body/shoulders
        this.createBody();
        
        this.scene.add(this.avatar);
    }
    
    createHead() {
        // Head geometry - more refined female shape
        const headGeometry = new THREE.SphereGeometry(1, 32, 32);
        
        // Modify geometry for more feminine features
        const positions = headGeometry.attributes.position.array;
        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i];
            const y = positions[i + 1];
            const z = positions[i + 2];
            
            // Create more oval shape
            positions[i] = x * 0.85;
            positions[i + 1] = y * 1.1;
            
            // Add subtle cheekbone definition
            if (y > 0.2 && y < 0.6 && Math.abs(x) > 0.3) {
                positions[i + 2] = z * 1.05;
            }
        }
        
        headGeometry.attributes.position.needsUpdate = true;
        headGeometry.computeVertexNormals();
        
        // Skin material with ethereal glow
        const skinMaterial = new THREE.MeshPhysicalMaterial({
            color: 0xffd4a3,
            roughness: 0.1,
            metalness: 0.1,
            clearcoat: 0.3,
            clearcoatRoughness: 0.1,
            transmission: 0.1,
            thickness: 0.5,
            emissive: 0x001122,
            emissiveIntensity: 0.1
        });
        
        const head = new THREE.Mesh(headGeometry, skinMaterial);
        head.castShadow = true;
        head.receiveShadow = true;
        head.position.y = 2;
        
        this.avatar.add(head);
        this.head = head;
    }
    
    createHair() {
        // Flowing hair with multiple layers
        const hairGroup = new THREE.Group();
        
        // Main hair volume
        const hairGeometry = new THREE.SphereGeometry(1.2, 16, 16);
        const hairMaterial = new THREE.MeshLambertMaterial({
            color: 0x8b4513,
            transparent: true,
            opacity: 0.9
        });
        
        const mainHair = new THREE.Mesh(hairGeometry, hairMaterial);
        mainHair.position.set(0, 2.3, -0.2);
        mainHair.scale.set(1, 0.8, 1.2);
        hairGroup.add(mainHair);
        
        // Hair strands
        for (let i = 0; i < 20; i++) {
            const strandGeometry = new THREE.CylinderGeometry(0.02, 0.01, 2, 4);
            const strand = new THREE.Mesh(strandGeometry, hairMaterial);
            
            const angle = (i / 20) * Math.PI * 2;
            strand.position.set(
                Math.cos(angle) * 0.8,
                1.5,
                Math.sin(angle) * 0.8 - 0.3
            );
            strand.rotation.z = Math.random() * 0.5 - 0.25;
            
            hairGroup.add(strand);
        }
        
        this.avatar.add(hairGroup);
        this.hair = hairGroup;
    }
    
    createEyes() {
        // Left eye
        const eyeGeometry = new THREE.SphereGeometry(0.15, 16, 16);
        const eyeMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x87ceeb,
            emissive: 0x001144,
            emissiveIntensity: 0.3,
            metalness: 0.1,
            roughness: 0.1
        });
        
        const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        leftEye.position.set(-0.25, 2.1, 0.7);
        
        const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
        rightEye.position.set(0.25, 2.1, 0.7);
        
        // Pupils
        const pupilGeometry = new THREE.SphereGeometry(0.08, 16, 16);
        const pupilMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
        
        const leftPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
        leftPupil.position.set(-0.25, 2.1, 0.78);
        
        const rightPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
        rightPupil.position.set(0.25, 2.1, 0.78);
        
        this.avatar.add(leftEye);
        this.avatar.add(rightEye);
        this.avatar.add(leftPupil);
        this.avatar.add(rightPupil);
        
        this.eyes = { left: leftEye, right: rightEye, leftPupil, rightPupil };
    }
    
    createAccessories() {
        // Greek goddess crown/diadem
        const crownGroup = new THREE.Group();
        
        // Main crown band
        const crownGeometry = new THREE.TorusGeometry(1.1, 0.05, 8, 32);
        const crownMaterial = new THREE.MeshPhysicalMaterial({
            color: 0xffd700,
            metalness: 0.9,
            roughness: 0.1,
            emissive: 0x221100,
            emissiveIntensity: 0.2
        });
        
        const crown = new THREE.Mesh(crownGeometry, crownMaterial);
        crown.position.set(0, 2.8, 0);
        crown.rotation.x = Math.PI / 2;
        crownGroup.add(crown);
        
        // Crown ornaments
        for (let i = 0; i < 8; i++) {
            const ornamentGeometry = new THREE.ConeGeometry(0.1, 0.3, 6);
            const ornament = new THREE.Mesh(ornamentGeometry, crownMaterial);
            
            const angle = (i / 8) * Math.PI * 2;
            ornament.position.set(
                Math.cos(angle) * 1.1,
                3,
                Math.sin(angle) * 1.1
            );
            
            crownGroup.add(ornament);
        }
        
        this.avatar.add(crownGroup);
        this.crown = crownGroup;
    }
    
    createBody() {
        // Shoulders and upper torso
        const shoulderGeometry = new THREE.CylinderGeometry(0.8, 1.2, 1.5, 16);
        const bodyMaterial = new THREE.MeshPhysicalMaterial({
            color: 0xffd4a3,
            roughness: 0.2,
            metalness: 0.1,
            clearcoat: 0.2
        });
        
        const shoulders = new THREE.Mesh(shoulderGeometry, bodyMaterial);
        shoulders.position.y = 0.5;
        shoulders.castShadow = true;
        shoulders.receiveShadow = true;
        
        // Greek dress/toga
        const dressGeometry = new THREE.ConeGeometry(1.5, 2, 16);
        const dressMaterial = new THREE.MeshPhysicalMaterial({
            color: 0xf0f8ff,
            roughness: 0.8,
            metalness: 0.1,
            transparent: true,
            opacity: 0.9,
            emissive: 0x001122,
            emissiveIntensity: 0.05
        });
        
        const dress = new THREE.Mesh(dressGeometry, dressMaterial);
        dress.position.y = -0.5;
        dress.castShadow = true;
        dress.receiveShadow = true;
        
        this.avatar.add(shoulders);
        this.avatar.add(dress);
        
        this.body = { shoulders, dress };
    }
    
    createPostProcessing() {
        this.composer = new EffectComposer(this.renderer);
        
        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);
        
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            0.5,    // strength
            0.4,    // radius
            0.85    // threshold
        );
        this.composer.addPass(bloomPass);
        
        this.bloomPass = bloomPass;
    }
    
    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
            this.composer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Control buttons
        document.getElementById('animate-btn').addEventListener('click', () => {
            this.toggleAnimation();
        });
        
        document.getElementById('expression-btn').addEventListener('click', () => {
            this.cycleExpression();
        });
        
        document.getElementById('lighting-btn').addEventListener('click', () => {
            this.cycleLighting();
        });
        
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetAvatar();
        });
    }
    
    toggleAnimation() {
        this.isAnimating = !this.isAnimating;
        const btn = document.getElementById('animate-btn');
        btn.textContent = this.isAnimating ? 'Stop' : 'Animate';
    }
    
    cycleExpression() {
        const expressions = ['neutral', 'smile', 'wise', 'powerful'];
        const currentIndex = expressions.indexOf(this.currentExpression);
        this.currentExpression = expressions[(currentIndex + 1) % expressions.length];
        
        // Animate facial expression changes
        this.animateExpression();
    }
    
    animateExpression() {
        // Simple expression animation by modifying eye and mouth positions
        switch (this.currentExpression) {
            case 'smile':
                // Slightly close eyes and adjust mouth
                this.eyes.left.scale.y = 0.8;
                this.eyes.right.scale.y = 0.8;
                break;
            case 'wise':
                // Slightly narrowed eyes
                this.eyes.left.scale.y = 0.9;
                this.eyes.right.scale.y = 0.9;
                this.head.rotation.x = -0.1;
                break;
            case 'powerful':
                // Wide eyes, head up
                this.eyes.left.scale.y = 1.2;
                this.eyes.right.scale.y = 1.2;
                this.head.rotation.x = 0.1;
                break;
            default:
                // Neutral
                this.eyes.left.scale.y = 1;
                this.eyes.right.scale.y = 1;
                this.head.rotation.x = 0;
        }
    }
    
    cycleLighting() {
        const modes = ['divine', 'ethereal', 'powerful', 'serene'];
        const currentIndex = modes.indexOf(this.lightingMode);
        this.lightingMode = modes[(currentIndex + 1) % modes.length];
        
        this.updateLighting();
    }
    
    updateLighting() {
        switch (this.lightingMode) {
            case 'ethereal':
                this.bloomPass.strength = 1.0;
                this.scene.children.forEach(child => {
                    if (child.type === 'DirectionalLight' && child.color.getHex() === 0x00ffff) {
                        child.color.setHex(0x9370db);
                    }
                });
                break;
            case 'powerful':
                this.bloomPass.strength = 1.5;
                this.scene.children.forEach(child => {
                    if (child.type === 'DirectionalLight' && child.color.getHex() !== 0xffffff) {
                        child.color.setHex(0xff4500);
                    }
                });
                break;
            case 'serene':
                this.bloomPass.strength = 0.3;
                this.scene.children.forEach(child => {
                    if (child.type === 'DirectionalLight' && child.color.getHex() !== 0xffffff) {
                        child.color.setHex(0x87ceeb);
                    }
                });
                break;
            default: // divine
                this.bloomPass.strength = 0.5;
                this.scene.children.forEach(child => {
                    if (child.type === 'DirectionalLight' && child.color.getHex() !== 0xffffff) {
                        child.color.setHex(0x00ffff);
                    }
                });
        }
    }
    
    resetAvatar() {
        // Reset camera position
        this.camera.position.set(0, 2, 8);
        this.controls.reset();
        
        // Reset avatar pose
        this.avatar.rotation.set(0, 0, 0);
        this.head.rotation.set(0, 0, 0);
        
        // Reset expression
        this.currentExpression = 'neutral';
        this.animateExpression();
        
        // Reset lighting
        this.lightingMode = 'divine';
        this.updateLighting();
        
        // Stop animation
        this.isAnimating = false;
        document.getElementById('animate-btn').textContent = 'Animate';
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const delta = this.clock.getDelta();
        
        // Update controls
        this.controls.update();
        
        // Animate particles
        if (this.animateParticles) {
            this.animateParticles();
        }
        
        // Avatar animations
        if (this.isAnimating && this.avatar) {
            // Gentle floating motion
            this.avatar.position.y = Math.sin(Date.now() * 0.001) * 0.1;
            
            // Subtle head movement
            this.head.rotation.y = Math.sin(Date.now() * 0.0005) * 0.1;
            
            // Hair movement
            if (this.hair) {
                this.hair.rotation.y = Math.sin(Date.now() * 0.0003) * 0.05;
            }
            
            // Crown glow
            if (this.crown) {
                this.crown.rotation.y += 0.005;
            }
            
            // Eye glow animation
            if (this.eyes) {
                const glowIntensity = (Math.sin(Date.now() * 0.002) + 1) * 0.15 + 0.1;
                this.eyes.left.material.emissiveIntensity = glowIntensity;
                this.eyes.right.material.emissiveIntensity = glowIntensity;
            }
        }
        
        // Render
        this.composer.render();
    }
}

// Initialize the avatar when the page loads
window.addEventListener('DOMContentLoaded', () => {
    new GreekGoddessAvatar();
});