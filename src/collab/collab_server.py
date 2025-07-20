import asyncio
import websockets
from typing import Dict, Set
import json
from src.agents.conflict_resolution import ConflictResolutionAgent  # Assume agent exists


class CollaborationServer:
    """
    Real-time collaboration server for multi-user sessions.
    Handles user authentication, session management, and message broadcasting.
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 8765):
        self.host = host
        self.port = port
        self.active_sessions = {}
        self.user_roles = {}  # session_id: {user_id: role}
        self.conflict_agent = ConflictResolutionAgent()

    async def handler(self, websocket, path):
        session_id = path.strip('/')
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = set()
            self.user_roles[session_id] = {}
        # Auth stub
        auth_msg = await websocket.recv()
        user_data = json.loads(auth_msg)
        user_id = user_data['user_id']
        role = user_data.get('role', 'editor')
        self.user_roles[session_id][user_id] = role
        self.active_sessions[session_id].add(websocket)
        await self.broadcast(session_id, json.dumps({'type': 'join', 'user': user_id}))
        try:
            async for message in websocket:
                data = json.loads(message)
                if data['type'] == 'edit':
                    if self.user_roles[session_id][user_id] in ['editor', 'admin']:
                        # Check for conflicts
                        resolved = await self.conflict_agent.resolve(data['edit'], session_id)
                        await self.broadcast(session_id, json.dumps({'type': 'edit', 'data': resolved}))
                    else:
                        await websocket.send(json.dumps({'type': 'error', 'msg': 'Permission denied'}))
                elif data['type'] == 'cursor':
                    await self.broadcast(session_id, message, exclude=websocket)
                else:
                    await self.broadcast(session_id, message)
        finally:
            self.active_sessions[session_id].remove(websocket)
            del self.user_roles[session_id][user_id]
            await self.broadcast(session_id, json.dumps({'type': 'leave', 'user': user_id}))

    async def broadcast(self, session_id: str, message: str, exclude=None):
        for ws in self.active_sessions.get(session_id, set()):
            if ws != exclude:
                await ws.send(message)

    def run(self):
        start_server = websockets.serve(self.handler, self.host, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        print(f"Collaboration server running on ws://{self.host}:{self.port}")
        asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    CollaborationServer().run() 