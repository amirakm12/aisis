#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <future>

namespace aisis {

struct Vertex {
    float position[3];
    float normal[3];
    float texCoords[2];
    float color[4];
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    GLuint VAO, VBO, EBO;
    GLuint textureID;
    std::string materialName;
};

struct Camera {
    float position[3]{0.0f, 0.0f, 5.0f};
    float target[3]{0.0f, 0.0f, 0.0f};
    float up[3]{0.0f, 1.0f, 0.0f};
    float fov{45.0f};
    float nearPlane{0.1f};
    float farPlane{1000.0f};
    float aspectRatio{16.0f/9.0f};
};

struct Light {
    enum Type { DIRECTIONAL, POINT, SPOT };
    Type type{POINT};
    float position[3]{0.0f, 5.0f, 0.0f};
    float direction[3]{0.0f, -1.0f, 0.0f};
    float color[3]{1.0f, 1.0f, 1.0f};
    float intensity{1.0f};
    float range{100.0f};
    float spotAngle{45.0f};
};

class RenderEngine {
public:
    RenderEngine();
    ~RenderEngine();
    
    // Initialization
    bool initialize(int windowWidth = 1920, int windowHeight = 1080);
    void shutdown();
    
    // Window management
    GLFWwindow* getWindow() const { return m_window; }
    void setWindowSize(int width, int height);
    void setFullscreen(bool fullscreen);
    void setVSync(bool enabled);
    
    // Rendering pipeline
    void beginFrame();
    void endFrame();
    void clear(float r = 0.1f, float g = 0.1f, float b = 0.1f, float a = 1.0f);
    
    // Mesh management
    uint32_t createMesh(const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices);
    void updateMesh(uint32_t meshId, const std::vector<Vertex>& vertices, const std::vector<unsigned int>& indices);
    void deleteMesh(uint32_t meshId);
    void renderMesh(uint32_t meshId, const float* modelMatrix);
    
    // Texture management
    uint32_t loadTexture(const std::string& path);
    uint32_t createTexture(int width, int height, const void* data, GLenum format = GL_RGBA);
    void bindTexture(uint32_t textureId, int slot = 0);
    void deleteTexture(uint32_t textureId);
    
    // Shader management
    uint32_t loadShader(const std::string& vertexPath, const std::string& fragmentPath);
    uint32_t createShaderFromSource(const std::string& vertexSource, const std::string& fragmentSource);
    void useShader(uint32_t shaderId);
    void deleteShader(uint32_t shaderId);
    void setShaderUniform(uint32_t shaderId, const std::string& name, const float* value, int count = 1);
    void setShaderUniform(uint32_t shaderId, const std::string& name, int value);
    void setShaderUniform(uint32_t shaderId, const std::string& name, float value);
    
    // Camera system
    void setCamera(const Camera& camera);
    Camera& getCamera() { return m_camera; }
    void updateCameraMatrices();
    
    // Lighting system
    uint32_t addLight(const Light& light);
    void updateLight(uint32_t lightId, const Light& light);
    void removeLight(uint32_t lightId);
    void setAmbientLight(float r, float g, float b, float intensity = 0.2f);
    
    // Post-processing effects
    void enableBloom(bool enable);
    void enableSSAO(bool enable);
    void enableHDR(bool enable);
    void enableAntiAliasing(bool enable, int samples = 4);
    void setToneMapping(int type); // 0=None, 1=Reinhard, 2=ACES, 3=Uncharted2
    
    // Performance features
    void enableGPUCulling(bool enable);
    void enableInstancedRendering(bool enable);
    void setLODSystem(bool enable);
    void enableAsyncResourceLoading(bool enable);
    
    // Image processing (using OpenCV)
    cv::Mat captureFramebuffer();
    void applyImageFilter(cv::Mat& image, const std::string& filterType);
    void applyComputeShader(uint32_t shaderId, int width, int height, int depth = 1);
    
    // Advanced rendering
    void renderSkybox(uint32_t cubemapId);
    void renderParticleSystem(const std::vector<float>& positions, const std::vector<float>& colors, float pointSize = 1.0f);
    void renderText(const std::string& text, float x, float y, float scale, const float* color);
    
    // Debug and profiling
    void enableWireframe(bool enable);
    void showFPS(bool show);
    void showRenderStats(bool show);
    float getFPS() const { return m_fps; }
    size_t getDrawCalls() const { return m_drawCalls; }
    size_t getTriangleCount() const { return m_triangleCount; }
    
    // Multi-threading support
    void enableMultiThreadedRendering(bool enable);
    void submitRenderCommand(std::function<void()> command);
    void flushRenderCommands();
    
private:
    // Core OpenGL objects
    GLFWwindow* m_window{nullptr};
    int m_windowWidth{1920};
    int m_windowHeight{1080};
    bool m_fullscreen{false};
    
    // Rendering state
    Camera m_camera;
    std::vector<Light> m_lights;
    uint32_t m_nextMeshId{1};
    uint32_t m_nextTextureId{1};
    uint32_t m_nextShaderId{1};
    uint32_t m_nextLightId{1};
    
    // Resource management
    std::unordered_map<uint32_t, std::unique_ptr<Mesh>> m_meshes;
    std::unordered_map<uint32_t, GLuint> m_textures;
    std::unordered_map<uint32_t, GLuint> m_shaders;
    
    // Performance tracking
    std::atomic<float> m_fps{0.0f};
    std::atomic<size_t> m_drawCalls{0};
    std::atomic<size_t> m_triangleCount{0};
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    
    // Post-processing
    GLuint m_framebuffer{0};
    GLuint m_colorTexture{0};
    GLuint m_depthTexture{0};
    bool m_bloomEnabled{false};
    bool m_ssaoEnabled{false};
    bool m_hdrEnabled{true};
    bool m_antiAliasingEnabled{true};
    int m_aaSamples{4};
    
    // Multi-threading
    bool m_multiThreadedRendering{false};
    std::vector<std::function<void()>> m_renderCommands;
    std::mutex m_commandMutex;
    std::thread m_renderThread;
    std::atomic<bool> m_renderThreadRunning{false};
    
    // Matrices
    float m_viewMatrix[16];
    float m_projectionMatrix[16];
    float m_viewProjectionMatrix[16];
    
    // Helper methods
    bool initializeOpenGL();
    bool createFramebuffers();
    void updatePerformanceStats();
    void renderThreadFunction();
    GLuint compileShader(const std::string& source, GLenum type);
    GLuint linkShaderProgram(GLuint vertexShader, GLuint fragmentShader);
    void calculateMatrices();
    
    // OpenCV integration
    cv::Mat m_frameBuffer;
    std::mutex m_frameMutex;
};

} // namespace aisis