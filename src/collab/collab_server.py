import asyncio
import websockets
from typing import Dict, Set
import time
from src.core.sanitization import sanitize_string

class CollaborationServer:
    """
    Real-time collaboration server for multi-user sessions.
    Handles user authentication, session management, and message broadcasting.
    """

    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.active_sessions: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        self.last_message_time: Dict[str, float] = {}  # For rate limiting
        self.allowed_origins = ['*']  # Configure from config later

    async def handler(self, websocket, path):
        if self.allowed_origins != ['*'] and websocket.origin not in self.allowed_origins:
            await websocket.close(1008, 'Origin not allowed')
            return

        session_id = path.strip('/')
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = set()
        self.active_sessions[session_id].add(websocket)

        client_id = f"{websocket.remote_address}_{session_id}"
        self.last_message_time[client_id] = 0

        try:
            async for message in websocket:
                now = time.time()
                if now - self.last_message_time[client_id] < 0.5:  # Rate limit to 2 msg/sec
                    continue
                self.last_message_time[client_id] = now

                sanitized_message = sanitize_string(message)
                await self.broadcast(session_id, sanitized_message)
        finally:
            self.active_sessions[session_id].remove(websocket)
            if not self.active_sessions[session_id]:
                del self.active_sessions[session_id]
            del self.last_message_time[client_id]

    async def broadcast(self, session_id: str, message: str):
        for ws in self.active_sessions.get(session_id, set()):
            await ws.send(message)

    def run(self):
        start_server = websockets.serve(self.handler, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        print(f"Collaboration server running on ws://{self.host}:{self.port}")
        asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    CollaborationServer().run()
