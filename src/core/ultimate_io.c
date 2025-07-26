#include "ultimate_core.h"
#include "ultimate_errors.h"
#include "ultimate_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif

/* File handle structure */
typedef struct {
    uint32_t id;
    ultimate_file_mode_t mode;
    const char* filename;
    bool is_open;
#ifdef _WIN32
    HANDLE handle;
#else
    int fd;
#endif
} ultimate_file_t;

/* Socket handle structure */
typedef struct {
    uint32_t id;
    ultimate_socket_type_t type;
    bool is_connected;
    bool is_listening;
    uint16_t local_port;
    uint16_t remote_port;
    char remote_address[16];
#ifdef _WIN32
    SOCKET socket;
#else
    int socket;
#endif
} ultimate_socket_t;

/* Global file and socket management */
static ultimate_file_t g_files[ULTIMATE_MAX_FILES];
static uint32_t g_file_count = 0;
static uint32_t g_next_file_id = 1;

static ultimate_socket_t g_sockets[ULTIMATE_MAX_SOCKETS];
static uint32_t g_socket_count = 0;
static uint32_t g_next_socket_id = 1;
static bool g_winsock_initialized = false;

/* File I/O Functions */
ultimate_error_t ultimate_file_open(const char* filename, ultimate_file_mode_t mode, 
                                   ultimate_file_handle_t* handle) {
    if (!filename || !handle || g_file_count >= ULTIMATE_MAX_FILES) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_file_t* file = &g_files[g_file_count];
    memset(file, 0, sizeof(ultimate_file_t));
    
    file->id = g_next_file_id++;
    file->mode = mode;
    file->filename = filename;
    
#ifdef _WIN32
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
    
    file->handle = CreateFileA(filename, access, FILE_SHARE_READ, NULL, 
                              creation, FILE_ATTRIBUTE_NORMAL, NULL);
    
    if (file->handle == INVALID_HANDLE_VALUE) {
        return ULTIMATE_ERROR_FILE_NOT_FOUND;
    }
    
    if (mode == ULTIMATE_FILE_MODE_APPEND) {
        SetFilePointer(file->handle, 0, NULL, FILE_END);
    }
#else
    int flags = 0;
    
    switch (mode) {
        case ULTIMATE_FILE_MODE_READ:
            flags = O_RDONLY;
            break;
        case ULTIMATE_FILE_MODE_WRITE:
            flags = O_WRONLY | O_CREAT | O_TRUNC;
            break;
        case ULTIMATE_FILE_MODE_APPEND:
            flags = O_WRONLY | O_CREAT | O_APPEND;
            break;
        case ULTIMATE_FILE_MODE_READ_WRITE:
            flags = O_RDWR | O_CREAT;
            break;
    }
    
    file->fd = open(filename, flags, 0644);
    if (file->fd < 0) {
        return ULTIMATE_ERROR_FILE_NOT_FOUND;
    }
#endif
    
    file->is_open = true;
    *handle = (ultimate_file_handle_t)file;
    g_file_count++;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_file_close(ultimate_file_handle_t handle) {
    ultimate_file_t* file = (ultimate_file_t*)handle;
    if (!file || !file->is_open) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    CloseHandle(file->handle);
#else
    close(file->fd);
#endif
    
    file->is_open = false;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_file_read(ultimate_file_handle_t handle, void* buffer, 
                                   size_t size, size_t* bytes_read) {
    ultimate_file_t* file = (ultimate_file_t*)handle;
    if (!file || !file->is_open || !buffer || !bytes_read) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    DWORD read;
    if (!ReadFile(file->handle, buffer, (DWORD)size, &read, NULL)) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
    *bytes_read = (size_t)read;
#else
    ssize_t result = read(file->fd, buffer, size);
    if (result < 0) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
    *bytes_read = (size_t)result;
#endif
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_file_write(ultimate_file_handle_t handle, const void* buffer, 
                                    size_t size, size_t* bytes_written) {
    ultimate_file_t* file = (ultimate_file_t*)handle;
    if (!file || !file->is_open || !buffer || !bytes_written) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    DWORD written;
    if (!WriteFile(file->handle, buffer, (DWORD)size, &written, NULL)) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
    *bytes_written = (size_t)written;
#else
    ssize_t result = write(file->fd, buffer, size);
    if (result < 0) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
    *bytes_written = (size_t)result;
#endif
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_file_seek(ultimate_file_handle_t handle, int64_t offset, int whence) {
    ultimate_file_t* file = (ultimate_file_t*)handle;
    if (!file || !file->is_open) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    DWORD move_method;
    switch (whence) {
        case 0: move_method = FILE_BEGIN; break;      // SEEK_SET
        case 1: move_method = FILE_CURRENT; break;    // SEEK_CUR
        case 2: move_method = FILE_END; break;        // SEEK_END
        default: return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    LARGE_INTEGER li;
    li.QuadPart = offset;
    
    if (SetFilePointer(file->handle, li.LowPart, &li.HighPart, move_method) == INVALID_SET_FILE_POINTER) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
#else
    if (lseek(file->fd, offset, whence) < 0) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
#endif
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_file_tell(ultimate_file_handle_t handle, int64_t* position) {
    ultimate_file_t* file = (ultimate_file_t*)handle;
    if (!file || !file->is_open || !position) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    LARGE_INTEGER li = {0};
    li.LowPart = SetFilePointer(file->handle, 0, &li.HighPart, FILE_CURRENT);
    if (li.LowPart == INVALID_SET_FILE_POINTER) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
    *position = li.QuadPart;
#else
    off_t pos = lseek(file->fd, 0, SEEK_CUR);
    if (pos < 0) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
    *position = (int64_t)pos;
#endif
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_file_flush(ultimate_file_handle_t handle) {
    ultimate_file_t* file = (ultimate_file_t*)handle;
    if (!file || !file->is_open) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    return FlushFileBuffers(file->handle) ? ULTIMATE_OK : ULTIMATE_ERROR_IO_ERROR;
#else
    return (fsync(file->fd) == 0) ? ULTIMATE_OK : ULTIMATE_ERROR_IO_ERROR;
#endif
}

ultimate_error_t ultimate_file_size(ultimate_file_handle_t handle, uint64_t* size) {
    ultimate_file_t* file = (ultimate_file_t*)handle;
    if (!file || !file->is_open || !size) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file->handle, &file_size)) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
    *size = (uint64_t)file_size.QuadPart;
#else
    struct stat st;
    if (fstat(file->fd, &st) < 0) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
    *size = (uint64_t)st.st_size;
#endif
    
    return ULTIMATE_OK;
}

bool ultimate_file_exists(const char* filename) {
    if (!filename) return false;
    
#ifdef _WIN32
    DWORD attrs = GetFileAttributesA(filename);
    return (attrs != INVALID_FILE_ATTRIBUTES);
#else
    return (access(filename, F_OK) == 0);
#endif
}

ultimate_error_t ultimate_file_delete(const char* filename) {
    if (!filename) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    return DeleteFileA(filename) ? ULTIMATE_OK : ULTIMATE_ERROR_IO_ERROR;
#else
    return (unlink(filename) == 0) ? ULTIMATE_OK : ULTIMATE_ERROR_IO_ERROR;
#endif
}

ultimate_error_t ultimate_file_copy(const char* source, const char* destination) {
    if (!source || !destination) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    return CopyFileA(source, destination, FALSE) ? ULTIMATE_OK : ULTIMATE_ERROR_IO_ERROR;
#else
    /* Simple file copy implementation for Linux */
    int src_fd = open(source, O_RDONLY);
    if (src_fd < 0) return ULTIMATE_ERROR_FILE_NOT_FOUND;
    
    int dst_fd = open(destination, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (dst_fd < 0) {
        close(src_fd);
        return ULTIMATE_ERROR_IO_ERROR;
    }
    
    char buffer[4096];
    ssize_t bytes;
    while ((bytes = read(src_fd, buffer, sizeof(buffer))) > 0) {
        if (write(dst_fd, buffer, bytes) != bytes) {
            close(src_fd);
            close(dst_fd);
            return ULTIMATE_ERROR_IO_ERROR;
        }
    }
    
    close(src_fd);
    close(dst_fd);
    return ULTIMATE_OK;
#endif
}

ultimate_error_t ultimate_file_move(const char* source, const char* destination) {
    if (!source || !destination) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    return MoveFileA(source, destination) ? ULTIMATE_OK : ULTIMATE_ERROR_IO_ERROR;
#else
    return (rename(source, destination) == 0) ? ULTIMATE_OK : ULTIMATE_ERROR_IO_ERROR;
#endif
}

/* Network/Socket Functions */
ultimate_error_t ultimate_network_init(void) {
#ifdef _WIN32
    if (!g_winsock_initialized) {
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (result != 0) {
            return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
        }
        g_winsock_initialized = true;
    }
#endif
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_network_cleanup(void) {
#ifdef _WIN32
    if (g_winsock_initialized) {
        WSACleanup();
        g_winsock_initialized = false;
    }
#endif
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_socket_create(ultimate_socket_type_t type, 
                                       ultimate_socket_handle_t* handle) {
    if (!handle || g_socket_count >= ULTIMATE_MAX_SOCKETS) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    ultimate_error_t error = ultimate_network_init();
    if (error != ULTIMATE_OK) {
        return error;
    }
    
    ultimate_socket_t* sock = &g_sockets[g_socket_count];
    memset(sock, 0, sizeof(ultimate_socket_t));
    
    sock->id = g_next_socket_id++;
    sock->type = type;
    
    int socket_type = (type == ULTIMATE_SOCKET_TYPE_TCP) ? SOCK_STREAM : SOCK_DGRAM;
    int protocol = (type == ULTIMATE_SOCKET_TYPE_TCP) ? IPPROTO_TCP : IPPROTO_UDP;
    
#ifdef _WIN32
    sock->socket = socket(AF_INET, socket_type, protocol);
    if (sock->socket == INVALID_SOCKET) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#else
    sock->socket = socket(AF_INET, socket_type, protocol);
    if (sock->socket < 0) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#endif
    
    *handle = (ultimate_socket_handle_t)sock;
    g_socket_count++;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_socket_bind(ultimate_socket_handle_t handle, 
                                     const char* address, uint16_t port) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    if (!sock || !address) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    
    if (strcmp(address, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        addr.sin_addr.s_addr = inet_addr(address);
    }
    
#ifdef _WIN32
    if (bind(sock->socket, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#else
    if (bind(sock->socket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#endif
    
    sock->local_port = port;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_socket_listen(ultimate_socket_handle_t handle, int backlog) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    if (!sock || sock->type != ULTIMATE_SOCKET_TYPE_TCP) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    if (listen(sock->socket, backlog) == SOCKET_ERROR) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#else
    if (listen(sock->socket, backlog) < 0) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#endif
    
    sock->is_listening = true;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_socket_accept(ultimate_socket_handle_t handle, 
                                       ultimate_socket_handle_t* client_handle) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    if (!sock || !client_handle || !sock->is_listening || g_socket_count >= ULTIMATE_MAX_SOCKETS) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    
#ifdef _WIN32
    SOCKET client_socket = accept(sock->socket, (struct sockaddr*)&client_addr, &addr_len);
    if (client_socket == INVALID_SOCKET) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#else
    int client_socket = accept(sock->socket, (struct sockaddr*)&client_addr, &addr_len);
    if (client_socket < 0) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#endif
    
    ultimate_socket_t* client_sock = &g_sockets[g_socket_count];
    memset(client_sock, 0, sizeof(ultimate_socket_t));
    
    client_sock->id = g_next_socket_id++;
    client_sock->type = ULTIMATE_SOCKET_TYPE_TCP;
    client_sock->socket = client_socket;
    client_sock->is_connected = true;
    client_sock->remote_port = ntohs(client_addr.sin_port);
    strcpy(client_sock->remote_address, inet_ntoa(client_addr.sin_addr));
    
    *client_handle = (ultimate_socket_handle_t)client_sock;
    g_socket_count++;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_socket_connect(ultimate_socket_handle_t handle, 
                                        const char* address, uint16_t port) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    if (!sock || !address) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(address);
    
#ifdef _WIN32
    if (connect(sock->socket, (struct sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#else
    if (connect(sock->socket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#endif
    
    sock->is_connected = true;
    sock->remote_port = port;
    strcpy(sock->remote_address, address);
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_socket_send(ultimate_socket_handle_t handle, 
                                     const void* data, size_t size, size_t* bytes_sent) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    if (!sock || !data || !bytes_sent) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    int result = send(sock->socket, (const char*)data, (int)size, 0);
    if (result == SOCKET_ERROR) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
#else
    ssize_t result = send(sock->socket, data, size, 0);
    if (result < 0) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
#endif
    
    *bytes_sent = (size_t)result;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_socket_receive(ultimate_socket_handle_t handle, 
                                        void* buffer, size_t size, size_t* bytes_received) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    if (!sock || !buffer || !bytes_received) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    int result = recv(sock->socket, (char*)buffer, (int)size, 0);
    if (result == SOCKET_ERROR) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
#else
    ssize_t result = recv(sock->socket, buffer, size, 0);
    if (result < 0) {
        return ULTIMATE_ERROR_IO_ERROR;
    }
#endif
    
    *bytes_received = (size_t)result;
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_socket_close(ultimate_socket_handle_t handle) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    if (!sock) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    closesocket(sock->socket);
#else
    close(sock->socket);
#endif
    
    sock->is_connected = false;
    sock->is_listening = false;
    
    return ULTIMATE_OK;
}

ultimate_error_t ultimate_socket_set_timeout(ultimate_socket_handle_t handle, 
                                            uint32_t timeout_ms) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    if (!sock) {
        return ULTIMATE_ERROR_INVALID_PARAMETER;
    }
    
#ifdef _WIN32
    DWORD timeout = timeout_ms;
    if (setsockopt(sock->socket, SOL_SOCKET, SO_RCVTIMEO, 
                   (const char*)&timeout, sizeof(timeout)) == SOCKET_ERROR) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    if (setsockopt(sock->socket, SOL_SOCKET, SO_SNDTIMEO, 
                   (const char*)&timeout, sizeof(timeout)) == SOCKET_ERROR) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    
    if (setsockopt(sock->socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
    if (setsockopt(sock->socket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) < 0) {
        return ULTIMATE_ERROR_SYSTEM_CALL_FAILED;
    }
#endif
    
    return ULTIMATE_OK;
}

bool ultimate_socket_is_connected(ultimate_socket_handle_t handle) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    return sock ? sock->is_connected : false;
}

uint16_t ultimate_socket_get_local_port(ultimate_socket_handle_t handle) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    return sock ? sock->local_port : 0;
}

uint16_t ultimate_socket_get_remote_port(ultimate_socket_handle_t handle) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    return sock ? sock->remote_port : 0;
}

const char* ultimate_socket_get_remote_address(ultimate_socket_handle_t handle) {
    ultimate_socket_t* sock = (ultimate_socket_t*)handle;
    return sock ? sock->remote_address : NULL;
}