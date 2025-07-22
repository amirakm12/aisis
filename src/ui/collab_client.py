import asyncio
import websockets
import json
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QApplication,
)
import sys
import threading
from collab import AutomergeCollab


class CollabClient(QObject):
    context_received = pyqtSignal(dict)
    log_received = pyqtSignal(str)
    presence_received = pyqtSignal(list)

    def __init__(self, url="ws://localhost:8765"):
        super().__init__()
        self.url = url
        self.ws = None
        self.loop = asyncio.get_event_loop()
        self.running = False

    async def connect(self):
        self.ws = await websockets.connect(self.url)
        self.running = True
        asyncio.create_task(self.listen())

    async def listen(self):
        while self.running:
            try:
                if self.ws is None:
                    break
                msg = await self.ws.recv()
                data = json.loads(msg)
                if data.get("type") == "context":
                    self.context_received.emit(data["context"])
                elif data.get("type") == "log":
                    self.log_received.emit(data["log"])
                elif data.get("type") == "presence":
                    self.presence_received.emit(data["users"])
            except Exception:
                break

    async def send_context(self, context):
        if self.ws:
            await self.ws.send(json.dumps({"type": "context", "context": context}))

    async def send_log(self, log):
        if self.ws:
            await self.ws.send(json.dumps({"type": "log", "log": log}))

    async def send_presence(self, users):
        if self.ws:
            await self.ws.send(json.dumps({"type": "presence", "users": users}))

    async def disconnect(self):
        self.running = False
        if self.ws:
            await self.ws.close()


class CollabClientPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Collaboration Panel")
        self.setMinimumWidth(350)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.status_label = QLabel("Not connected")
        self.layout.addWidget(self.status_label)

        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Relay server IP (e.g., 192.168.1.10)")
        self.layout.addWidget(self.ip_input)

        self.port_input = QLineEdit()
        self.port_input.setPlaceholderText("Port (default: 9009)")
        self.layout.addWidget(self.port_input)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_to_relay)
        self.layout.addWidget(self.connect_btn)

        self.collab = None
        self.connected = False

    def connect_to_relay(self):
        ip = self.ip_input.text().strip()
        port = self.port_input.text().strip()
        if not ip:
            QMessageBox.warning(self, "Input Error", "Please enter the relay server IP.")
            return
        try:
            port = int(port) if port else 9009
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Port must be a number.")
            return
        self.status_label.setText("Connecting...")
        self.collab = AutomergeCollab(doc_id="default", on_remote_change=self.on_remote_change)
        try:
            threading.Thread(
                target=self.collab.connect_to_relay, args=(ip, port), daemon=True
            ).start()
            self.status_label.setText(f"Connected to {ip}:{port}")
            self.connected = True
        except Exception as e:
            self.status_label.setText("Connection failed")
            QMessageBox.critical(self, "Connection Error", str(e))
            self.connected = False

    def on_remote_change(self):
        # This method is called when a remote change is received
        self.status_label.setText("Received update from collaborator!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    panel = CollabClientPanel()
    panel.show()
    sys.exit(app.exec_())
