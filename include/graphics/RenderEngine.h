#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>

namespace Ultimate {
namespace Graphics {

// Forward declarations
class Shader;
class Texture;
class Mesh;
class Camera;
class Light;
class Material;

// Rendering API types
enum class RenderAPI {
    OpenGL,
    Vulkan,
    DirectX11,
    DirectX12,
    Metal
};

// Render modes
enum class RenderMode {
    Forward,
    Deferred,
    ForwardPlus,
    Hybrid
};

// Quality settings
enum class QualityLevel {
    Low = 1,
    Medium = 2,
    High = 3,
    Ultra = 4,
    Extreme = 5
};

struct RenderStats {
    int drawCalls = 0;
    int triangles = 0;
    int vertices = 0;
    double frameTime = 0.0;
    double cpuTime = 0.0;
    double gpuTime = 0.0;
    size_t vramUsage = 0;
};

class RenderEngine {
public:
    static RenderEngine& getInstance();
    
    // Initialization and cleanup
    bool initialize(RenderAPI api = RenderAPI::OpenGL);
    void shutdown();
    
    // Rendering pipeline
    void beginFrame();
    void endFrame();
    void present();
    
    // Scene management
    void setActiveCamera(std::shared_ptr<Camera> camera);
    std::shared_ptr<Camera> getActiveCamera() const;
    
    void addLight(std::shared_ptr<Light> light);
    void removeLight(std::shared_ptr<Light> light);
    void clearLights();
    
    // Rendering
    void renderMesh(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material);
    void renderMeshInstanced(std::shared_ptr<Mesh> mesh, std::shared_ptr<Material> material, 
                           const std::vector<glm::mat4>& transforms);
    
    // Post-processing
    void enablePostProcessing(bool enable);
    bool isPostProcessingEnabled() const;
    
    void addPostProcessEffect(const std::string& name, std::shared_ptr<Shader> shader);
    void removePostProcessEffect(const std::string& name);
    void setPostProcessOrder(const std::vector<std::string>& order);
    
    // Quality and performance
    void setQualityLevel(QualityLevel level);
    QualityLevel getQualityLevel() const;
    
    void setRenderMode(RenderMode mode);
    RenderMode getRenderMode() const;
    
    // Advanced features
    void enableHDR(bool enable);
    bool isHDREnabled() const;
    
    void enableMSAA(int samples);
    int getMSAASamples() const;
    
    void enableVSync(bool enable);
    bool isVSyncEnabled() const;
    
    void enableRayTracing(bool enable);
    bool isRayTracingEnabled() const;
    
    void enableVariableRateShading(bool enable);
    bool isVariableRateShadingEnabled() const;
    
    // Shadows
    void enableShadows(bool enable);
    bool areShadowsEnabled() const;
    
    void setShadowQuality(QualityLevel quality);
    QualityLevel getShadowQuality() const;
    
    void setShadowDistance(float distance);
    float getShadowDistance() const;
    
    // Lighting
    void setAmbientLight(const glm::vec3& color);
    glm::vec3 getAmbientLight() const;
    
    void enableGlobalIllumination(bool enable);
    bool isGlobalIlluminationEnabled() const;
    
    // Debug and profiling
    const RenderStats& getRenderStats() const;
    void resetRenderStats();
    
    void enableWireframe(bool enable);
    bool isWireframeEnabled() const;
    
    void enableDebugOverlay(bool enable);
    bool isDebugOverlayEnabled() const;
    
    // Resource management
    std::shared_ptr<Shader> createShader(const std::string& vertexSource, 
                                       const std::string& fragmentSource);
    std::shared_ptr<Texture> createTexture(const std::string& filepath);
    std::shared_ptr<Mesh> createMesh(const std::vector<float>& vertices, 
                                   const std::vector<unsigned int>& indices);
    
    // Viewport and resolution
    void setViewport(int x, int y, int width, int height);
    void getViewport(int& x, int& y, int& width, int& height) const;
    
    void setResolution(int width, int height);
    void getResolution(int& width, int& height) const;
    
    // Callbacks
    using RenderCallback = std::function<void()>;
    void setPreRenderCallback(RenderCallback callback);
    void setPostRenderCallback(RenderCallback callback);

private:
    RenderEngine() = default;
    ~RenderEngine() = default;
    RenderEngine(const RenderEngine&) = delete;
    RenderEngine& operator=(const RenderEngine&) = delete;
    
    // Internal state
    bool m_initialized = false;
    RenderAPI m_api = RenderAPI::OpenGL;
    RenderMode m_renderMode = RenderMode::Forward;
    QualityLevel m_qualityLevel = QualityLevel::High;
    
    // Scene data
    std::shared_ptr<Camera> m_activeCamera;
    std::vector<std::shared_ptr<Light>> m_lights;
    
    // Post-processing
    bool m_postProcessingEnabled = true;
    std::unordered_map<std::string, std::shared_ptr<Shader>> m_postProcessEffects;
    std::vector<std::string> m_postProcessOrder;
    
    // Settings
    bool m_hdrEnabled = false;
    int m_msaaSamples = 1;
    bool m_vsyncEnabled = true;
    bool m_rayTracingEnabled = false;
    bool m_variableRateShadingEnabled = false;
    bool m_shadowsEnabled = true;
    QualityLevel m_shadowQuality = QualityLevel::High;
    float m_shadowDistance = 100.0f;
    glm::vec3 m_ambientLight{0.1f, 0.1f, 0.1f};
    bool m_globalIlluminationEnabled = false;
    
    // Debug
    bool m_wireframeEnabled = false;
    bool m_debugOverlayEnabled = false;
    
    // Viewport
    int m_viewportX = 0, m_viewportY = 0;
    int m_viewportWidth = 1920, m_viewportHeight = 1080;
    
    // Stats
    RenderStats m_renderStats;
    
    // Callbacks
    RenderCallback m_preRenderCallback;
    RenderCallback m_postRenderCallback;
    
    // Internal methods
    void setupDefaultState();
    void applyQualitySettings();
    void updateRenderStats();
    void executePostProcessing();
};

} // namespace Graphics
} // namespace Ultimate