# GUI Research for Aisis Image Restoration App

## Overview
Researched modern UI/UX trends for image editing/restoration apps (e.g., Photoshop, Lightroom, Affinity Photo, GIMP 3.0 betas, AI tools like Midjourney web UI, Runway ML, Stable Diffusion webUIs). Focus on revolutionary features: intuitive workflows, AI-assisted interfaces, dark/light modes, responsive design, gesture support.

## Key Ideas & Inspirations
1. **Modular Dashboard**: Sidebar with agent selectors (e.g., dropdown for restoration models like RAIM, DarkIR). Central canvas for image preview with zoom/pan. Bottom timeline for edit history/undo.
   - Inspiration: Adobe Lightroom's module system.

2. **AI-Powered Suggestions**: On image upload, auto-analyze and suggest agents (e.g., "Detected low-light: Try DarkIR"). Floating AI chat for natural language commands.
   - Inspiration: Midjourney's prompt-based interface + ChatGPT integration in Canva.

3. **Before/After Split View**: Split-screen comparison with slider, real-time previews during processing.
   - Inspiration: Photoshop's history states and Lightroom's compare view.

4. **Dark Mode & Themes**: Auto dark/light mode, customizable themes for accessibility.
   - Inspiration: Figma, VS Code themes.

5. **Touch/Gesture Support**: For tablet/mobile, pinch-zoom, swipe to switch agents, shake to undo.
   - Inspiration: Procreate app.

6. **Progress & Feedback**: Animated progress bars, confidence scores, error tooltips. Visual diff maps showing changes.
   - Inspiration: GitHub's code diff viewer adapted to images.

7. **Collaboration Features**: Real-time multi-user editing, version control like Google Docs for images.
   - Inspiration: Figma's collaborative design.

8. **VR/AR Preview**: Optional AR mode to preview restorations on real-world objects via camera.
   - Inspiration: IKEA Place app.

## Implementation Plan
- Use Qt for cross-platform (existing base).
- Integrate Qt Quick/QML for modern, responsive UI.
- Add PySide6 for Python bindings.
- Structure: MainWindow with QStackedWidget for modules, QGraphicsView for canvas.
- Revolutionary Twist: "Agent Flow" graph view to chain models visually (like Node-RED for AI pipelines).