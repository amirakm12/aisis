#include <stdio.h>
#include <stdlib.h>
#include "aisis.h"

/**
 * @brief Main entry point of the AISIS application
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Exit status
 */
int main(int argc, char *argv[]) {
    printf("AISIS Application Starting...\n");
    printf("Version: 1.0.0\n");
    printf("Build: %s %s\n", __DATE__, __TIME__);
    
    // Initialize AISIS library
    if (aisis_init() != 0) {
        fprintf(stderr, "Error: Failed to initialize AISIS library\n");
        return EXIT_FAILURE;
    }
    
    printf("AISIS library initialized successfully\n");
    
    // Main application logic
    int result = aisis_run();
    if (result != 0) {
        fprintf(stderr, "Error: AISIS execution failed with code %d\n", result);
    }
    
    // Cleanup
    aisis_cleanup();
    
    printf("AISIS Application Exiting...\n");
    return result;
}