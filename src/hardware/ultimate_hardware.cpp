#include "ultimate_hardware.h"
#include "ultimate_errors.h"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mmsystem.h>
#include <dsound.h>
#include <d3d11.h>
#include <dxgi.h>
#include <winreg.h>
#include <winsvc.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "dsound.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "advapi32.lib")
#endif

namespace ultimate {

struct WindowContext {
    ultimate_window_handle_t handle;
    HWND hwnd;
    ultimate_window_config_t config;
    ultimate_input_callback_t input_callback;
    void* user_data;
    bool visible;
    
    WindowContext() : handle(0), hwnd(nullptr), input_callback(nullptr), user_data(nullptr), visible(false) {}
};

struct FileContext {
    ultimate_file_handle_t handle;
    HANDLE file_handle;
    ultimate_file_mode_t mode;
    
    FileContext() : handle(0), file_handle(INVALID_HANDLE_VALUE), mode(ULTIMATE_FILE_MODE_READ) {}
};

struct SocketContext {
    ultimate_socket_handle_t handle;
    SOCKET socket;
    ultimate_socket_type_t type;
    bool connected;
    
    SocketContext() : handle(0), socket(INVALID_SOCKET), type(ULTIMATE_SOCKET_TYPE_TCP), connected(false) {}
};

class HardwareManager {
private:
    static std::unique_ptr<HardwareManager> instance_;
    static std::mutex instance_mutex_;
    
    std::mutex hardware_mutex_;
    std::unordered_map<ultimate_window_handle_t, std::unique_ptr<WindowContext>> windows_;
    std::unordered_map<ultimate_file_handle_t, std::unique_ptr<FileContext>> files_;
    std::unordered_map<ultimate_socket_handle_t, std::unique_ptr<SocketContext>> sockets_;
    
    std::atomic<ultimate_window_handle_t> next_window_handle_;
    std::atomic<ultimate_file_handle_t> next_file_handle_;
    std::atomic<ultimate_socket_handle_t> next_socket_handle_;
    
    bool initialized_;
    HINSTANCE hinstance_;
    WNDCLASS window_class_;
    bool winsock_initialized_;
    
    HardwareManager() : next_window_handle_(1), next_file_handle_(1), next_socket_handle_(1),
                       initialized_(false), hinstance_(nullptr), winsock_initialized_(false) {}

public:
    static HardwareManager* getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = std::unique_ptr<HardwareManager>(new HardwareManager());
        }
        return instance_.get();
    }
    
    ultimate_error_t initialize() {
        if (initialized_) {
            return ULTIMATE_ERROR_ALREADY_INITIALIZED;
        }
        
#ifdef _WIN32
        hinstance_ = GetModuleHandle(nullptr);
        
        // Register window class
        window_class_.style = CS_HREDRAW | CS_VREDRAW;
        window_class_.lpfnWndProc = windowProc;
        window_class_.cbClsExtra = 0;
        window_class_.cbWndExtra = 0;
        window_class_.hInstance = hinstance_;
        window_class_.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
        window_class_.hCursor = LoadCursor(nullptr, IDC_ARROW);
        window_class_.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        window_class_.lpszMenuName = nullptr;
        window_class_.lpszClassName = L"UltimateWindow";
        
        if (!RegisterClass(&window_class_)) {
            return ULTIMATE_ERROR_INITIALIZATION_FAILED;
        }
        
        // Initialize Winsock
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            return ULTIMATE_ERROR_NETWORK_INITIALIZATION_FAILED;
        }
        winsock_initialized_ = true;
#endif
        
        initialized_ = true;
        return ULTIMATE_OK;
    }
    
    ultimate_error_t createWindow(const ultimate_window_config_t* config,
                                 ultimate_window_handle_t* handle) {
        if (!initialized_ || !config || !handle) {
            return ULTIMATE_ERROR_INVALID_PARAMETER;
        }
        
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
#ifdef _WIN32
        auto window = std::make_unique<WindowContext>();
        window->handle = next_window_handle_.fetch_add(1);
        window->config = *config;
        
        DWORD style = WS_OVERLAPPEDWINDOW;
        if (!config->resizable) {
            style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        }
        
        if (config->fullscreen) {
            style = WS_POPUP;
        }
        
        // Convert title to wide string
        int title_len = MultiByteToWideChar(CP_UTF8, 0, config->title, -1, nullptr, 0);
        std::vector<wchar_t> wide_title(title_len);
        MultiByteToWideChar(CP_UTF8, 0, config->title, -1, wide_title.data(), title_len);
        
        window->hwnd = CreateWindow(
            L"UltimateWindow",
            wide_title.data(),
            style,
            CW_USEDEFAULT, CW_USEDEFAULT,
            config->width, config->height,
            nullptr, nullptr,
            hinstance_,
            window.get()
        );
        
        if (!window->hwnd) {
            return ULTIMATE_ERROR_WINDOW_CREATION_FAILED;
        }
        
        *handle = window->handle;
        windows_[window->handle] = std::move(window);
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t showWindow(ultimate_window_handle_t handle) {
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        auto it = windows_.find(handle);
        if (it == windows_.end()) {
            return ULTIMATE_ERROR_WINDOW_NOT_FOUND;
        }
        
#ifdef _WIN32
        auto& window = it->second;
        ShowWindow(window->hwnd, SW_SHOW);
        UpdateWindow(window->hwnd);
        window->visible = true;
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t hideWindow(ultimate_window_handle_t handle) {
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        auto it = windows_.find(handle);
        if (it == windows_.end()) {
            return ULTIMATE_ERROR_WINDOW_NOT_FOUND;
        }
        
#ifdef _WIN32
        auto& window = it->second;
        ShowWindow(window->hwnd, SW_HIDE);
        window->visible = false;
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t registerInputCallback(ultimate_window_handle_t handle,
                                          ultimate_input_callback_t callback,
                                          void* user_data) {
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        auto it = windows_.find(handle);
        if (it == windows_.end()) {
            return ULTIMATE_ERROR_WINDOW_NOT_FOUND;
        }
        
        auto& window = it->second;
        window->input_callback = callback;
        window->user_data = user_data;
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t openFile(const char* file_path, ultimate_file_mode_t mode,
                             ultimate_file_handle_t* handle) {
        if (!initialized_ || !file_path || !handle) {
            return ULTIMATE_ERROR_INVALID_PARAMETER;
        }
        
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
#ifdef _WIN32
        auto file = std::make_unique<FileContext>();
        file->handle = next_file_handle_.fetch_add(1);
        file->mode = mode;
        
        DWORD access = 0;
        DWORD creation = 0;
        
        switch (mode) {
            case ULTIMATE_FILE_MODE_READ:
                access = GENERIC_READ;
                creation = OPEN_EXISTING;
                break;
            case ULTIMATE_FILE_MODE_WRITE:
                access = GENERIC_WRITE;
                creation = CREATE_ALWAYS;
                break;
            case ULTIMATE_FILE_MODE_APPEND:
                access = GENERIC_WRITE;
                creation = OPEN_ALWAYS;
                break;
            case ULTIMATE_FILE_MODE_READ_WRITE:
                access = GENERIC_READ | GENERIC_WRITE;
                creation = OPEN_ALWAYS;
                break;
        }
        
        // Convert path to wide string
        int path_len = MultiByteToWideChar(CP_UTF8, 0, file_path, -1, nullptr, 0);
        std::vector<wchar_t> wide_path(path_len);
        MultiByteToWideChar(CP_UTF8, 0, file_path, -1, wide_path.data(), path_len);
        
        file->file_handle = CreateFile(
            wide_path.data(),
            access,
            FILE_SHARE_READ,
            nullptr,
            creation,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        
        if (file->file_handle == INVALID_HANDLE_VALUE) {
            return ULTIMATE_ERROR_FILE_OPEN_FAILED;
        }
        
        // For append mode, seek to end
        if (mode == ULTIMATE_FILE_MODE_APPEND) {
            SetFilePointer(file->file_handle, 0, nullptr, FILE_END);
        }
        
        *handle = file->handle;
        files_[file->handle] = std::move(file);
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t readFile(ultimate_file_handle_t handle, void* buffer,
                             size_t buffer_size, size_t* bytes_read) {
        if (!initialized_ || !buffer || !bytes_read) {
            return ULTIMATE_ERROR_INVALID_PARAMETER;
        }
        
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        auto it = files_.find(handle);
        if (it == files_.end()) {
            return ULTIMATE_ERROR_FILE_NOT_FOUND;
        }
        
#ifdef _WIN32
        auto& file = it->second;
        
        DWORD read = 0;
        if (!ReadFile(file->file_handle, buffer, static_cast<DWORD>(buffer_size), &read, nullptr)) {
            return ULTIMATE_ERROR_FILE_READ_FAILED;
        }
        
        *bytes_read = read;
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t writeFile(ultimate_file_handle_t handle, const void* data,
                              size_t data_size, size_t* bytes_written) {
        if (!initialized_ || !data || !bytes_written) {
            return ULTIMATE_ERROR_INVALID_PARAMETER;
        }
        
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        auto it = files_.find(handle);
        if (it == files_.end()) {
            return ULTIMATE_ERROR_FILE_NOT_FOUND;
        }
        
#ifdef _WIN32
        auto& file = it->second;
        
        DWORD written = 0;
        if (!WriteFile(file->file_handle, data, static_cast<DWORD>(data_size), &written, nullptr)) {
            return ULTIMATE_ERROR_FILE_WRITE_FAILED;
        }
        
        *bytes_written = written;
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t closeFile(ultimate_file_handle_t handle) {
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        auto it = files_.find(handle);
        if (it == files_.end()) {
            return ULTIMATE_ERROR_FILE_NOT_FOUND;
        }
        
#ifdef _WIN32
        auto& file = it->second;
        if (file->file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(file->file_handle);
        }
#endif
        
        files_.erase(it);
        return ULTIMATE_OK;
    }
    
    ultimate_error_t createSocket(ultimate_socket_type_t type, ultimate_socket_handle_t* handle) {
        if (!initialized_ || !handle) {
            return ULTIMATE_ERROR_INVALID_PARAMETER;
        }
        
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
#ifdef _WIN32
        auto socket_ctx = std::make_unique<SocketContext>();
        socket_ctx->handle = next_socket_handle_.fetch_add(1);
        socket_ctx->type = type;
        
        int socket_type = (type == ULTIMATE_SOCKET_TYPE_TCP) ? SOCK_STREAM : SOCK_DGRAM;
        int protocol = (type == ULTIMATE_SOCKET_TYPE_TCP) ? IPPROTO_TCP : IPPROTO_UDP;
        
        socket_ctx->socket = socket(AF_INET, socket_type, protocol);
        if (socket_ctx->socket == INVALID_SOCKET) {
            return ULTIMATE_ERROR_SOCKET_CREATION_FAILED;
        }
        
        *handle = socket_ctx->handle;
        sockets_[socket_ctx->handle] = std::move(socket_ctx);
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t connectSocket(ultimate_socket_handle_t handle, const char* address, uint16_t port) {
        if (!initialized_ || !address) {
            return ULTIMATE_ERROR_INVALID_PARAMETER;
        }
        
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        auto it = sockets_.find(handle);
        if (it == sockets_.end()) {
            return ULTIMATE_ERROR_SOCKET_NOT_FOUND;
        }
        
#ifdef _WIN32
        auto& socket_ctx = it->second;
        
        sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        
        if (inet_pton(AF_INET, address, &addr.sin_addr) <= 0) {
            return ULTIMATE_ERROR_INVALID_ADDRESS;
        }
        
        if (connect(socket_ctx->socket, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
            return ULTIMATE_ERROR_SOCKET_CONNECT_FAILED;
        }
        
        socket_ctx->connected = true;
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t sendSocket(ultimate_socket_handle_t handle, const void* data,
                               size_t data_size, size_t* bytes_sent) {
        if (!initialized_ || !data || !bytes_sent) {
            return ULTIMATE_ERROR_INVALID_PARAMETER;
        }
        
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        auto it = sockets_.find(handle);
        if (it == sockets_.end()) {
            return ULTIMATE_ERROR_SOCKET_NOT_FOUND;
        }
        
#ifdef _WIN32
        auto& socket_ctx = it->second;
        
        int sent = send(socket_ctx->socket, static_cast<const char*>(data), static_cast<int>(data_size), 0);
        if (sent == SOCKET_ERROR) {
            return ULTIMATE_ERROR_SOCKET_SEND_FAILED;
        }
        
        *bytes_sent = sent;
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t receiveSocket(ultimate_socket_handle_t handle, void* buffer,
                                  size_t buffer_size, size_t* bytes_received) {
        if (!initialized_ || !buffer || !bytes_received) {
            return ULTIMATE_ERROR_INVALID_PARAMETER;
        }
        
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        auto it = sockets_.find(handle);
        if (it == sockets_.end()) {
            return ULTIMATE_ERROR_SOCKET_NOT_FOUND;
        }
        
#ifdef _WIN32
        auto& socket_ctx = it->second;
        
        int received = recv(socket_ctx->socket, static_cast<char*>(buffer), static_cast<int>(buffer_size), 0);
        if (received == SOCKET_ERROR) {
            return ULTIMATE_ERROR_SOCKET_RECEIVE_FAILED;
        }
        
        *bytes_received = received;
#endif
        
        return ULTIMATE_OK;
    }
    
    ultimate_error_t closeSocket(ultimate_socket_handle_t handle) {
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        auto it = sockets_.find(handle);
        if (it == sockets_.end()) {
            return ULTIMATE_ERROR_SOCKET_NOT_FOUND;
        }
        
#ifdef _WIN32
        auto& socket_ctx = it->second;
        if (socket_ctx->socket != INVALID_SOCKET) {
            closesocket(socket_ctx->socket);
        }
#endif
        
        sockets_.erase(it);
        return ULTIMATE_OK;
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(hardware_mutex_);
        
        // Close all windows
        for (auto& pair : windows_) {
#ifdef _WIN32
            if (pair.second->hwnd) {
                DestroyWindow(pair.second->hwnd);
            }
#endif
        }
        windows_.clear();
        
        // Close all files
        for (auto& pair : files_) {
#ifdef _WIN32
            if (pair.second->file_handle != INVALID_HANDLE_VALUE) {
                CloseHandle(pair.second->file_handle);
            }
#endif
        }
        files_.clear();
        
        // Close all sockets
        for (auto& pair : sockets_) {
#ifdef _WIN32
            if (pair.second->socket != INVALID_SOCKET) {
                closesocket(pair.second->socket);
            }
#endif
        }
        sockets_.clear();
        
#ifdef _WIN32
        if (winsock_initialized_) {
            WSACleanup();
            winsock_initialized_ = false;
        }
        
        UnregisterClass(L"UltimateWindow", hinstance_);
#endif
        
        initialized_ = false;
    }

private:
#ifdef _WIN32
    static LRESULT CALLBACK windowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        WindowContext* window = nullptr;
        
        if (msg == WM_NCCREATE) {
            CREATESTRUCT* cs = reinterpret_cast<CREATESTRUCT*>(lParam);
            window = static_cast<WindowContext*>(cs->lpCreateParams);
            SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(window));
        } else {
            window = reinterpret_cast<WindowContext*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
        }
        
        if (window && window->input_callback) {
            ultimate_input_event_t event = {};
            
            switch (msg) {
                case WM_KEYDOWN:
                case WM_KEYUP:
                    event.type = ULTIMATE_INPUT_TYPE_KEYBOARD;
                    event.key_code = static_cast<uint32_t>(wParam);
                    event.pressed = (msg == WM_KEYDOWN);
                    window->input_callback(window->handle, &event, window->user_data);
                    break;
                    
                case WM_LBUTTONDOWN:
                case WM_LBUTTONUP:
                case WM_RBUTTONDOWN:
                case WM_RBUTTONUP:
                case WM_MBUTTONDOWN:
                case WM_MBUTTONUP:
                    event.type = ULTIMATE_INPUT_TYPE_MOUSE;
                    event.mouse_x = LOWORD(lParam);
                    event.mouse_y = HIWORD(lParam);
                    event.pressed = (msg == WM_LBUTTONDOWN || msg == WM_RBUTTONDOWN || msg == WM_MBUTTONDOWN);
                    
                    if (msg == WM_LBUTTONDOWN || msg == WM_LBUTTONUP) {
                        event.mouse_button = ULTIMATE_MOUSE_BUTTON_LEFT;
                    } else if (msg == WM_RBUTTONDOWN || msg == WM_RBUTTONUP) {
                        event.mouse_button = ULTIMATE_MOUSE_BUTTON_RIGHT;
                    } else {
                        event.mouse_button = ULTIMATE_MOUSE_BUTTON_MIDDLE;
                    }
                    
                    window->input_callback(window->handle, &event, window->user_data);
                    break;
                    
                case WM_MOUSEMOVE:
                    event.type = ULTIMATE_INPUT_TYPE_MOUSE;
                    event.mouse_x = LOWORD(lParam);
                    event.mouse_y = HIWORD(lParam);
                    event.pressed = false;
                    window->input_callback(window->handle, &event, window->user_data);
                    break;
            }
        }
        
        switch (msg) {
            case WM_DESTROY:
                PostQuitMessage(0);
                break;
            default:
                return DefWindowProc(hwnd, msg, wParam, lParam);
        }
        
        return 0;
    }
#endif
};

std::unique_ptr<HardwareManager> HardwareManager::instance_ = nullptr;
std::mutex HardwareManager::instance_mutex_;

} // namespace ultimate

// C API Implementation
extern "C" {

ultimate_error_t ultimate_hardware_init(void) {
    return ultimate::HardwareManager::getInstance()->initialize();
}

ultimate_error_t ultimate_window_create(const ultimate_window_config_t* config,
                                       ultimate_window_handle_t* handle) {
    return ultimate::HardwareManager::getInstance()->createWindow(config, handle);
}

ultimate_error_t ultimate_window_show(ultimate_window_handle_t handle) {
    return ultimate::HardwareManager::getInstance()->showWindow(handle);
}

ultimate_error_t ultimate_window_hide(ultimate_window_handle_t handle) {
    return ultimate::HardwareManager::getInstance()->hideWindow(handle);
}

ultimate_error_t ultimate_input_register_callback(ultimate_window_handle_t handle,
                                                 ultimate_input_callback_t callback,
                                                 void* user_data) {
    return ultimate::HardwareManager::getInstance()->registerInputCallback(handle, callback, user_data);
}

ultimate_error_t ultimate_file_open(const char* file_path, ultimate_file_mode_t mode,
                                   ultimate_file_handle_t* handle) {
    return ultimate::HardwareManager::getInstance()->openFile(file_path, mode, handle);
}

ultimate_error_t ultimate_file_read(ultimate_file_handle_t handle, void* buffer,
                                   size_t buffer_size, size_t* bytes_read) {
    return ultimate::HardwareManager::getInstance()->readFile(handle, buffer, buffer_size, bytes_read);
}

ultimate_error_t ultimate_file_write(ultimate_file_handle_t handle, const void* data,
                                    size_t data_size, size_t* bytes_written) {
    return ultimate::HardwareManager::getInstance()->writeFile(handle, data, data_size, bytes_written);
}

ultimate_error_t ultimate_file_close(ultimate_file_handle_t handle) {
    return ultimate::HardwareManager::getInstance()->closeFile(handle);
}

ultimate_error_t ultimate_socket_create(ultimate_socket_type_t type, ultimate_socket_handle_t* handle) {
    return ultimate::HardwareManager::getInstance()->createSocket(type, handle);
}

ultimate_error_t ultimate_socket_connect(ultimate_socket_handle_t handle, const char* address, uint16_t port) {
    return ultimate::HardwareManager::getInstance()->connectSocket(handle, address, port);
}

ultimate_error_t ultimate_socket_send(ultimate_socket_handle_t handle, const void* data,
                                     size_t data_size, size_t* bytes_sent) {
    return ultimate::HardwareManager::getInstance()->sendSocket(handle, data, data_size, bytes_sent);
}

ultimate_error_t ultimate_socket_receive(ultimate_socket_handle_t handle, void* buffer,
                                        size_t buffer_size, size_t* bytes_received) {
    return ultimate::HardwareManager::getInstance()->receiveSocket(handle, buffer, buffer_size, bytes_received);
}

ultimate_error_t ultimate_socket_close(ultimate_socket_handle_t handle) {
    return ultimate::HardwareManager::getInstance()->closeSocket(handle);
}

void ultimate_hardware_deinit(void) {
    ultimate::HardwareManager::getInstance()->cleanup();
}

} // extern "C"