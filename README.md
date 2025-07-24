# Greek Goddess Avatar - 3D Interactive App

A stunning 3D Greek goddess avatar built with Three.js, featuring ethereal lighting, particle effects, and interactive controls.

## Features

✨ **3D Greek Goddess Avatar** - Beautifully crafted with:
- Detailed facial features with glowing eyes
- Flowing hair with individual strands
- Golden crown with ornamental details
- Elegant Greek dress/toga
- Realistic skin materials with subsurface scattering

🌟 **Interactive Controls**:
- **Drag to rotate** - Orbit around the goddess
- **Scroll to zoom** - Get closer or step back
- **Animate button** - Toggle floating animation with subtle movements
- **Expression button** - Cycle through different facial expressions (neutral, smile, wise, powerful)
- **Lighting button** - Switch between lighting modes (divine, ethereal, powerful, serene)
- **Reset button** - Return to default position and settings

🎨 **Visual Effects**:
- Bloom post-processing for ethereal glow
- 1000+ animated particles creating a magical atmosphere
- Dynamic lighting with multiple colored lights
- Shadows and reflections
- Atmospheric fog

## Quick Start

### Option 1: Using Node.js and Vite (Recommended)

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Open your browser** and navigate to `http://localhost:3000`

### Option 2: Using Python (Simple HTTP Server)

1. **Start a simple HTTP server:**
   ```bash
   npm run serve
   ```

2. **Open your browser** and navigate to `http://localhost:8000`

### Option 3: Direct Browser (with CORS limitations)

Simply open `index.html` in your browser. Note: Some features may not work due to CORS restrictions.

## Controls

- **Mouse Drag**: Rotate camera around the goddess
- **Mouse Wheel**: Zoom in/out
- **Animate**: Toggle breathing and floating animation
- **Expression**: Change facial expressions
- **Lighting**: Switch between different lighting moods
- **Reset**: Return to default view

## Technical Details

- **Three.js**: 3D graphics and rendering
- **WebGL**: Hardware-accelerated graphics
- **Post-processing**: Bloom effects for ethereal glow
- **Particle System**: 1000+ animated particles
- **PBR Materials**: Physically-based rendering for realistic materials
- **Shadow Mapping**: Real-time shadows
- **Orbit Controls**: Smooth camera interaction

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

Requires WebGL support.

## Customization

The avatar can be easily customized by modifying the `js/main.js` file:

- **Colors**: Change material colors in the create methods
- **Lighting**: Adjust light positions and intensities
- **Animations**: Modify the animation parameters
- **Expressions**: Add new facial expressions
- **Accessories**: Add more Greek goddess elements

## Performance

The application is optimized for modern browsers and should run smoothly on most devices. For better performance on lower-end devices:

- Reduce particle count in `createParticles()`
- Lower shadow map resolution
- Disable post-processing effects

## License

MIT License - Feel free to use and modify for your projects.

## Credits

Created with Three.js and modern web technologies. Inspired by classical Greek art and mythology.