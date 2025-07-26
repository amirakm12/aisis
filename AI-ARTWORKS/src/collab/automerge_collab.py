import automerge
import threading
import socket
import pickle
from typing import Callable, Optional

class AutomergeCollab:
    def __init__(self, doc_id: str, on_remote_change: Optional[Callable] = None):
        self.doc_id = doc_id
        self.doc = automerge.Document()
        self.lock = threading.Lock()
        self.relay_host = None
        self.relay_port = None
        self.relay_sock = None
        self.on_remote_change = on_remote_change  # Callback for UI updates
        self.listening = False
        self.peers = []  # List of (ip, port) tuples

    def apply_local_change(self, change_func):
        with self.lock:
            self.doc = change_func(self.doc)
            # After local change, send to relay
            self.send_change_to_relay()

    def get_state(self):
        with self.lock:
            return self.doc.save()

    def load_state(self, state_bytes):
        with self.lock:
            self.doc = automerge.Document.load(state_bytes)
        if self.on_remote_change:
            self.on_remote_change()

    def connect_to_relay(self, host: str, port: int):
        self.relay_host = host
        self.relay_port = port
        self.relay_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.relay_sock.connect((host, port))
        self.listening = True
        threading.Thread(target=self.listen_to_relay, daemon=True).start()

    def send_change_to_relay(self):
        if self.relay_sock:
            try:
                state = self.get_state()
                data = pickle.dumps(state)
                self.relay_sock.sendall(data)
            except Exception as e:
                print(f"Failed to send change to relay: {e}")

    def listen_to_relay(self):
        while self.listening:
            try:
                data = self.relay_sock.recv(4096)
                if data:
                    try:
                        state = pickle.loads(data)
                        self.load_state(state)
                    except Exception as e:
                        print(f"Error loading state from relay: {e}")
            except Exception as e:
                print(f"Relay connection error: {e}")
                self.listening = False
                break

    def close(self):
        self.listening = False
        if self.relay_sock:
            try:
                self.relay_sock.close()
            except Exception:
                pass

    def add_peer(self, ip, port):
        self.peers.append((ip, port))

    def broadcast_change(self, change_bytes):
        for ip, port in self.peers:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((ip, port))
                    s.sendall(change_bytes)
            except Exception as e:
                print(f"Failed to send to {ip}:{port}: {e}")

    def listen_for_changes(self, host='0.0.0.0', port=9009):
        def _listen():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                s.listen()
                while True:
                    conn, addr = s.accept()
                    with conn:
                        data = conn.recv(4096)
                        if data:
                            self.handle_incoming_change(data)
        thread = threading.Thread(target=_listen, daemon=True)
        thread.start()

    def handle_incoming_change(self, change_bytes):
        try:
            state = pickle.loads(change_bytes)
            self.load_state(state)
        except Exception as e:
            print(f"Error handling incoming change: {e}") 