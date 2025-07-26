import socket
import threading

class LANRelayServer:
    def __init__(self, host='0.0.0.0', port=9009):
        self.host = host
        self.port = port
        self.clients = []  # List of (conn, addr)
        self.lock = threading.Lock()

    def start(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen()
        print(f"LAN Relay Server listening on {self.host}:{self.port}")
        threading.Thread(target=self.accept_clients, args=(server_sock,), daemon=True).start()
        try:
            while True:
                pass  # Keep main thread alive
        except KeyboardInterrupt:
            print("Shutting down relay server.")
            server_sock.close()

    def accept_clients(self, server_sock):
        while True:
            conn, addr = server_sock.accept()
            print(f"Client connected: {addr}")
            with self.lock:
                self.clients.append((conn, addr))
            threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()

    def handle_client(self, conn, addr):
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                self.broadcast(data, exclude=conn)
        except Exception as e:
            print(f"Error with client {addr}: {e}")
        finally:
            print(f"Client disconnected: {addr}")
            with self.lock:
                self.clients = [(c, a) for c, a in self.clients if c != conn]
            conn.close()

    def broadcast(self, data, exclude=None):
        with self.lock:
            for conn, addr in self.clients:
                if conn != exclude:
                    try:
                        conn.sendall(data)
                    except Exception as e:
                        print(f"Failed to send to {addr}: {e}")

if __name__ == "__main__":
    server = LANRelayServer()
    server.start() 