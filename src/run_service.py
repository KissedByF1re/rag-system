import asyncio
import sys
import os
import argparse
import logging
import time
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from core import settings

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('service')


def main():
    parser = argparse.ArgumentParser(description='Run the Agent Service')
    parser.add_argument('--rebuild-db', action='store_true', help='Rebuild the database schema from scratch')
    parser.add_argument('--hot-reload', action='store_true', help='Enable enhanced hot reload with watchdog')
    args = parser.parse_args()
    
    # Check environment mode - consider both settings and command line args
    is_dev_mode = settings.is_dev() or os.getenv("DEV_MODE", "").lower() == "true"
    
    # Enhanced hot reload override from command line
    enhanced_hot_reload = args.hot_reload
    
    # Check if we're running in a Docker environment
    in_docker = os.path.exists("/.dockerenv")
    logger.info(f"Running in Docker environment: {in_docker}")
    
    # Enable hot-reload in dev mode, regardless of Docker or local
    if is_dev_mode:
        reload_enabled = True
        # Directories to watch for changes - include more directories
        reload_dirs = ["agents", "core", "memory", "schema", "service", "data_store"]
        logger.info(f"Running in DEVELOPMENT mode with hot reloading for directories: {reload_dirs}")
        
        # If enhanced hot reload is enabled, set up additional watch options
        if enhanced_hot_reload:
            logger.info("Enhanced hot reload enabled!")
            os.environ["WATCHFILES_FORCE_POLLING"] = "true"  # Force polling in Docker
    else:
        reload_enabled = False
        reload_dirs = None
        logger.info("Running in PRODUCTION mode (hot reloading disabled)")
    
    logger.info(f"Starting Agent Service on {settings.HOST}:{settings.PORT}")
    # Log Phoenix support status from environment variable
    phoenix_env = os.getenv("PHOENIX_ENABLED", "false").lower() == "true"
    logger.info(f"Phoenix support enabled (from env var PHOENIX_ENABLED): {phoenix_env}")
    # Verify Phoenix support flag in the service module
    try:
        from service.service import PHOENIX_ENABLED as phoenix_active
        logger.info(f"Phoenix support active in service module: {phoenix_active}")
    except Exception:
        logger.warning("Could not verify PHOENIX_ENABLED in service.service module")
    
    # Prepare uvicorn run arguments
    uvicorn_kwargs = {
        # host and port
        "host": settings.HOST,
        "port": settings.PORT,
        # whether to enable uvicorn auto-reload
        "reload": reload_enabled,
    }
    
    # Uvicorn requires the app parameter as first positional argument
    uvicorn_app = "service.service:app"
    
    # Add reload directories and includes only if reload is enabled
    if reload_enabled:
        # specify directories to watch for changes
        uvicorn_kwargs["reload_dirs"] = reload_dirs
        # watch only Python files
        uvicorn_kwargs["reload_includes"] = ["*.py"]
        # decrease delay between reload checks for faster response
        uvicorn_kwargs["reload_delay"] = 0.1
        
        # Use better settings for hot reload in Docker
        if enhanced_hot_reload:
            # Configure environment variables for better file watching in containers
            os.environ["WATCHFILES_FORCE_POLLING"] = "true"
            os.environ["WATCHFILES_POLL_INTERVAL"] = "500"
            logger.info("Using enhanced polling-based file monitoring for Docker")
    
    # Run the ASGI server
    uvicorn.run(uvicorn_app, **uvicorn_kwargs)

if __name__ == "__main__":
    # Set Compatible event loop policy on Windows Systems.
    # On Windows systems, the default ProactorEventLoop can cause issues with
    # certain async database drivers like psycopg (PostgreSQL driver).
    # The WindowsSelectorEventLoopPolicy provides better compatibility and prevents
    # "RuntimeError: Event loop is closed" errors when working with database connections.
    # This needs to be set before running the application server.
    # Refer to the documentation for more information.
    # https://www.psycopg.org/psycopg3/docs/advanced/async.html#asynchronous-operations
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
