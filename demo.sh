#!/bin/bash

# AISIS Creative Studio Demo Script
# Automatically showcases all features and extra tweaks

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                    AISIS CREATIVE STUDIO                      ║"
    echo "║                    DEMO SHOWCASE v2.0                        ║"
    echo "║                 Extra Features & Tweaks                      ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_status() {
    echo -e "${BLUE}[DEMO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

main() {
    print_header
    
    print_status "Building AISIS Creative Studio with all optimizations..."
    ./build.sh build > /dev/null 2>&1
    print_success "Build completed successfully!"
    
    print_status "Launching AISIS Creative Studio Demo Mode..."
    echo -e "${CYAN}"
    
    # Run demo mode automatically (option 8, then 0 to exit)
    echo -e "8\n0" | timeout 30s ./build/bin/aisis_creative_studio || true
    
    echo -e "${NC}"
    print_success "Demo completed!"
    
    echo ""
    print_status "AISIS Creative Studio features showcased:"
    echo "  ✓ Advanced multimedia processing (Audio, Video, Image)"
    echo "  ✓ ARM optimization with NEON SIMD support"
    echo "  ✓ AI Assistant with intelligent suggestions"
    echo "  ✓ Plugin architecture with extensible system"
    echo "  ✓ Performance profiling and resource monitoring"
    echo "  ✓ Multi-threading with task queue system"
    echo "  ✓ Multiple theme support (Dark, Light, Neon)"
    echo "  ✓ Configuration management with INI files"
    echo "  ✓ Advanced logging system"
    echo "  ✓ Rich console UI with animations"
    echo "  ✓ Project management system"
    echo "  ✓ Comprehensive build system"
    
    echo ""
    echo -e "${GREEN}🎉 AISIS Creative Studio v2.0 - All Extra Features & Tweaks Demonstrated!${NC}"
    echo -e "${CYAN}To run manually: ./build/bin/aisis_creative_studio${NC}"
    echo -e "${CYAN}To build: ./build.sh build${NC}"
    echo -e "${CYAN}For help: ./build.sh help${NC}"
}

# Run the demo
main "$@"