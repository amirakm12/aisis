#!/usr/bin/env python3
"""
Startup script for AI Content-Aware Storage with RAG
This script helps with initial setup and environment validation.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def check_environment_file():
    """Check if .env file exists and has required variables"""
    env_file = Path('.env')
    if not env_file.exists():
        logger.warning(".env file not found. Creating from .env.example...")
        example_file = Path('.env.example')
        if example_file.exists():
            import shutil
            shutil.copy('.env.example', '.env')
            logger.info("Created .env file from .env.example")
            logger.warning("Please edit .env file with your configuration before starting the application")
            return False
        else:
            logger.error(".env.example file not found")
            return False
    
    # Check for required environment variables
    required_vars = [
        'OPENAI_API_KEY',
        'DATABASE_URL',
        'REDIS_URL',
        'SECRET_KEY'
    ]
    
    missing_vars = []
    with open('.env', 'r') as f:
        env_content = f.read()
        for var in required_vars:
            if f"{var}=" not in env_content or f"{var}=your_" in env_content:
                missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing or incomplete environment variables: {', '.join(missing_vars)}")
        logger.error("Please update your .env file with proper values")
        return False
    
    logger.info("Environment configuration looks good")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'chroma_db', 'static', 'templates']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import langchain
        import chromadb
        import openai
        import sqlalchemy
        import redis
        logger.info("All major dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return False

def check_services():
    """Check if required services are running"""
    services_ok = True
    
    # Check Redis
    try:
        import redis
        r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
        r.ping()
        logger.info("Redis connection: OK")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        logger.warning("Please ensure Redis is running")
        services_ok = False
    
    # Check PostgreSQL
    try:
        import sqlalchemy
        from app.config import settings
        engine = sqlalchemy.create_engine(settings.database_url)
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        logger.info("PostgreSQL connection: OK")
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed: {e}")
        logger.warning("Please ensure PostgreSQL is running and database exists")
        services_ok = False
    
    return services_ok

def install_dependencies():
    """Install dependencies from requirements.txt"""
    try:
        logger.info("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def main():
    """Main startup function"""
    logger.info("🤖 AI Content-Aware Storage with RAG - Startup Script")
    logger.info("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create necessary directories
    create_directories()
    
    # Check if dependencies are installed
    if not check_dependencies():
        logger.info("Attempting to install dependencies...")
        if not install_dependencies():
            sys.exit(1)
    
    # Check environment configuration
    if not check_environment_file():
        logger.error("Please configure your .env file and run the script again")
        sys.exit(1)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed, environment variables may not be loaded")
    
    # Check services
    services_ok = check_services()
    
    if services_ok:
        logger.info("✅ All checks passed! Starting the application...")
        logger.info("=" * 60)
        
        # Start the application
        try:
            import uvicorn
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
                log_level="info"
            )
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            sys.exit(1)
    else:
        logger.warning("⚠️  Some services are not available, but you can still try to start the application")
        logger.info("Run 'python main.py' to start the application")

if __name__ == "__main__":
    main()