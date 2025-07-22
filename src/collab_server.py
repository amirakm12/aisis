import asyncio
import websockets
import json

clients = set()


async def handler(websocket, path):
    clients.add(websocket)
    try:
        async for message in websocket:
            # Broadcast received message to all clients
            for client in clients:
                if client != websocket:
                    await client.send(message)
    finally:
        clients.remove(websocket)


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Collaboration server running on ws://localhost:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
