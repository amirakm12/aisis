import asyncio
import websockets
from typing import Dict, Set


class CollaborationServer:
    """
    Real-time collaboration server for multi-user sessions.
    Handles user authentication, session management, and message broadcasting.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.active_sessions: Dict[
            str, Set[websockets.WebSocketServerProtocol]
        ] = {}

    async def handler(self, websocket, path):
        # TODO: Authenticate user, join session, handle messages
        session_id = path.strip("/")
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = set()
        self.active_sessions[session_id].add(websocket)
        try:
            async for message in websocket:
                await self.broadcast(session_id, message)
        finally:
            self.active_sessions[session_id].remove(websocket)

    async def broadcast(self, session_id: str, message: str):
        # Broadcast message to all users in the session
        for ws in self.active_sessions.get(session_id, set()):
            await ws.send(message)

    def run(self):
        start_server = websockets.serve(self.handler, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        print(f"Collaboration server running on ws://{self.host}:{self.port}")
        asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    CollaborationServer().run()
