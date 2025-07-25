"""
Al-artworks REST API Server
Comprehensive API for all Al-artworks functionality
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import shutil
from pathlib import Path
import asyncio
import uuid
from datetime import datetime

from . import alartworks, __version__
from src.core.config import config
from loguru import logger

# Pydantic models for API
class ProcessingRequest(BaseModel):
    operations: Optional[List[str]] = None
    agent: Optional[str] = None
    quality: str = "medium"
    parameters: Optional[Dict[str, Any]] = None

class ProcessingResponse(BaseModel):
    job_id: str
    status: str
    message: str
    result_url: Optional[str] = None

class AgentInfo(BaseModel):
    name: str
    description: str
    capabilities: List[str]
    model: str
    parameters: Dict[str, Any]

class ModelInfo(BaseModel):
    name: str
    version: str
    size: str
    status: str
    download_url: Optional[str] = None

class SystemStatus(BaseModel):
    version: str
    status: str
    gpu_available: bool
    memory_usage: Dict[str, float]
    active_jobs: int
    uptime: str

class ConfigUpdate(BaseModel):
    key: str
    value: Any

# Global state
processing_jobs = {}
app = FastAPI(
    title="Al-artworks API",
    description="AI Creative Studio REST API",
    version=__version__
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize Al-artworks on startup"""
    try:
        alartworks.initialize()
        logger.info("Al-artworks API server started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Al-artworks: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    alartworks.shutdown()
    logger.info("Al-artworks API server shutdown complete")

@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint"""
    return {
        "name": "Al-artworks API",
        "version": __version__,
        "description": "AI Creative Studio REST API",
        "docs": "/docs"
    }

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    try:
        device_manager = alartworks.device_manager
        device_info = device_manager.get_device_info()
        
        return SystemStatus(
            version=__version__,
            status="operational",
            gpu_available=device_info.get("gpu_available", False),
            memory_usage=device_info.get("memory_usage", {}),
            active_jobs=len(processing_jobs),
            uptime=str(datetime.now())  # Simplified uptime
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=ProcessingResponse)
async def process_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    operations: Optional[str] = None,
    agent: Optional[str] = None,
    quality: str = "medium"
):
    """Process an uploaded image"""
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        temp_dir = Path(tempfile.gettempdir()) / "al-artworks" / job_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = temp_dir / file.filename
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse operations
        ops_list = operations.split(",") if operations else None
        
        # Start background processing
        processing_jobs[job_id] = {
            "status": "processing",
            "created_at": datetime.now(),
            "input_path": str(input_path),
            "output_path": None,
            "error": None
        }
        
        background_tasks.add_task(
            process_image_task,
            job_id,
            str(input_path),
            ops_list,
            agent,
            quality
        )
        
        return ProcessingResponse(
            job_id=job_id,
            status="processing",
            message="Image processing started"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_image_task(job_id: str, input_path: str, operations: Optional[List[str]], 
                           agent: Optional[str], quality: str):
    """Background task for image processing"""
    try:
        # Set quality preset
        config.set_quality_preset(quality)
        
        # Process image
        result = alartworks.process_image(
            image_path=input_path,
            operations=operations,
            agent=agent
        )
        
        # Save result
        output_path = Path(input_path).parent / "result.jpg"
        result.save(str(output_path))
        
        # Update job status
        processing_jobs[job_id].update({
            "status": "completed",
            "output_path": str(output_path),
            "completed_at": datetime.now()
        })
        
    except Exception as e:
        processing_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now()
        })

@app.get("/jobs/{job_id}", response_model=ProcessingResponse)
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    result_url = None
    if job["status"] == "completed" and job["output_path"]:
        result_url = f"/results/{job_id}"
    
    return ProcessingResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("error", "Processing complete" if job["status"] == "completed" else "Processing..."),
        result_url=result_url
    )

@app.get("/results/{job_id}")
async def get_result(job_id: str):
    """Download processing result"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed" or not job["output_path"]:
        raise HTTPException(status_code=404, detail="Result not available")
    
    return FileResponse(
        job["output_path"],
        media_type="image/jpeg",
        filename=f"result_{job_id}.jpg"
    )

@app.get("/agents", response_model=List[AgentInfo])
async def list_agents():
    """List available agents"""
    try:
        agents = alartworks.get_available_agents()
        
        return [
            AgentInfo(
                name=name,
                description=info.get("description", ""),
                capabilities=info.get("capabilities", []),
                model=info.get("model", "unknown"),
                parameters=info.get("parameters", {})
            )
            for name, info in agents.items()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models"""
    try:
        models = alartworks.model_manager.list_models()
        
        return [
            ModelInfo(
                name=model["name"],
                version=model.get("version", "unknown"),
                size=model.get("size", "unknown"),
                status=model.get("status", "unknown"),
                download_url=model.get("download_url")
            )
            for model in models
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/download")
async def download_model(model_name: str, background_tasks: BackgroundTasks):
    """Download a specific model"""
    try:
        background_tasks.add_task(
            alartworks.model_manager.download_model,
            model_name
        )
        
        return {"message": f"Download started for model: {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """Get current configuration"""
    try:
        return config.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
async def update_config(config_update: ConfigUpdate):
    """Update configuration"""
    try:
        config.set(config_update.key, config_update.value)
        config.save()
        
        return {"message": f"Configuration updated: {config_update.key}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plugins")
async def list_plugins():
    """List installed plugins"""
    try:
        plugins = alartworks.plugin_manager.list_plugins()
        return plugins
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plugins/{plugin_name}/enable")
async def enable_plugin(plugin_name: str):
    """Enable a plugin"""
    try:
        alartworks.plugin_manager.enable_plugin(plugin_name)
        return {"message": f"Plugin '{plugin_name}' enabled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/plugins/{plugin_name}/disable")
async def disable_plugin(plugin_name: str):
    """Disable a plugin"""
    try:
        alartworks.plugin_manager.disable_plugin(plugin_name)
        return {"message": f"Plugin '{plugin_name}' disabled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark")
async def run_benchmark():
    """Run system benchmark"""
    try:
        from src.core.model_benchmarking import ModelBenchmarker
        
        benchmarker = ModelBenchmarker()
        results = benchmarker.run_benchmarks()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a processing job and its files"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        job = processing_jobs[job_id]
        
        # Clean up files
        if job.get("input_path"):
            input_path = Path(job["input_path"])
            if input_path.exists():
                shutil.rmtree(input_path.parent, ignore_errors=True)
        
        # Remove from jobs
        del processing_jobs[job_id]
        
        return {"message": f"Job {job_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "alartworks.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )