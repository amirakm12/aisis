import psutil
import aiohttp
from aiohttp import web
import asyncio
import json

async def health(request):
    return web.Response(text="OK")

async def metrics(request):
    data = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
    }
    return web.json_response(data)

async def start_monitoring_server():
    app = web.Application()
    app.router.add_get('/health', health)
    app.router.add_get('/metrics', metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    print("Monitoring server running on http://0.0.0.0:8080")

if __name__ == "__main__":
    asyncio.run(start_monitoring_server())
