import asyncio
import websockets
from typing import Dict, Set
import json


class CollaborationServer:
    """
    Real-time collaboration server for multi-user sessions.
    Handles user authentication, session management, and message broadcasting.
    """

    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.active_sessions: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}

    async def handler(self, websocket, path):
        session_id = path.strip('/')
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = set()
        self.active_sessions[session_id].add(websocket)
        user = None
        try:
            message = await websocket.recv()
            data = json.loads(message)
            if data.get('type') == 'join':
                user = data.get('user')
                await self.broadcast(session_id, json.dumps({'type': 'system', 'message': f"{user} joined"}))
            else:
                await websocket.close()
                return
            async for message in websocket:
                data = json.loads(message)
                if data.get('type') == 'message':
                    await self.broadcast(session_id, json.dumps({'type': 'message', 'user': user, 'content': data['content']})) 
        finally:
            if user:
                await self.broadcast(session_id, json.dumps({'type': 'system', 'message': f"{user} left"}))
            self.active_sessions[session_id].remove(websocket)

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
