#include <iostream>
#include <vector>
#include <string>

// Example function to demonstrate debugging
int calculate(int a, int b) {
    return a + b;
}

// Example class
class Application {
private:
    std::string name;
    std::vector<int> data;

public:
    Application(const std::string& appName) : name(appName) {
        std::cout << "Application '" << name << "' initialized." << std::endl;
    }

    void addData(int value) {
        data.push_back(value);
        std::cout << "Added value: " << value << std::endl;
    }

    void processData() {
        std::cout << "Processing " << data.size() << " data points..." << std::endl;
        
        int sum = 0;
        for (const auto& val : data) {
            sum = calculate(sum, val);
        }
        
        std::cout << "Sum of all data: " << sum << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "AISIS Application Starting..." << std::endl;
    
    // Print command line arguments if any
    if (argc > 1) {
        std::cout << "Command line arguments:" << std::endl;
        for (int i = 1; i < argc; ++i) {
            std::cout << "  Arg[" << i << "]: " << argv[i] << std::endl;
        }
    }
    
    // Create and use application
    Application app("AISIS");
    
    // Add some test data
    app.addData(10);
    app.addData(20);
    app.addData(30);
    
    // Process the data
    app.processData();
    
    std::cout << "Application finished successfully!" << std::endl;
    return 0;
}