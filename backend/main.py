from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import asyncio

from config import Settings
from database import init_db, engine
from models.device import Base

# Import routers
from routers import devices

# Load settings
settings = Settings()
APP_STATE = {
    "database_ready": False,
    "database_error": None,
}


def _get_nlp_status() -> dict:
    """Return NLP loading state without forcing model initialization."""
    try:
        from ml.nlp_service import get_nlp_service

        nlp_service = get_nlp_service()
        if not nlp_service.enable_heavy_nlp:
            state = "disabled"
        elif nlp_service.is_loaded:
            state = "loaded"
        elif nlp_service.initialization_in_progress:
            state = "loading"
        elif nlp_service.load_error:
            state = "error"
        else:
            state = "idle"

        return {
            "state": state,
            "heavy_nlp_enabled": nlp_service.enable_heavy_nlp,
            "is_loaded": nlp_service.is_loaded,
            "initialization_attempted": nlp_service.initialization_attempted,
            "initialization_in_progress": nlp_service.initialization_in_progress,
            "load_error": nlp_service.load_error,
        }
    except Exception as e:
        return {
            "state": "unavailable",
            "heavy_nlp_enabled": False,
            "is_loaded": False,
            "initialization_attempted": False,
            "initialization_in_progress": False,
            "load_error": f"NLP status unavailable: {str(e)}",
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database connection on startup and close on shutdown"""
    # Startup
    try:
        # Do not let slow DB init block app startup indefinitely on free tiers.
        await asyncio.wait_for(asyncio.to_thread(init_db), timeout=20)
        APP_STATE["database_ready"] = True
        APP_STATE["database_error"] = None
        print(f"[OK] Connected to MySQL database: {settings.database_name}")
    except asyncio.TimeoutError:
        APP_STATE["database_ready"] = False
        APP_STATE["database_error"] = "Database init timeout"
        print("[WARN] Database initialization timed out; starting API in degraded mode")
    except Exception as e:
        APP_STATE["database_ready"] = False
        APP_STATE["database_error"] = str(e)
        print(f"[WARN] Database initialization failed; starting API in degraded mode: {str(e)}")

    # Keep startup fast in production platforms (Render port scan timeout).
    # NLP models are still lazy-loaded on first request by the router.
    preload_nlp = settings.preload_nlp_on_startup
    if preload_nlp:
        async def _preload_nlp_background():
            try:
                print("[INFO] Pre-loading NLP models in background...")
                from routers.devices import nlp_service_preload
                await asyncio.to_thread(nlp_service_preload)
                print("[OK] NLP models pre-loaded successfully")
            except Exception as e:
                print(f"[WARN] NLP pre-loading skipped: {str(e)}")
                print("   (NLP will lazy-load on first request)")

        asyncio.create_task(_preload_nlp_background())
    else:
        print("[INFO] NLP pre-loading disabled for non-development environment")
    
    yield
    
    # Shutdown
    engine.dispose()
    print("[CLOSE] Closed MySQL connection")


# Create FastAPI app
app = FastAPI(
    title="SmartAI Device Filter API",
    description="AI-powered smart device recommendation and filtering system",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = settings.allowed_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=settings.allowed_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress API responses to reduce bandwidth and time-to-first-byte on free tiers.
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Include routers
app.include_router(devices.router, prefix="/api/devices", tags=["Devices"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SmartAI Device Filter API",
        "version": "2.0.0",
        "status": "running",
        "database": settings.database_name
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "database": "mysql",
        "ready": True,
        "database_ready": APP_STATE["database_ready"],
        "database_error": APP_STATE["database_error"],
        "nlp": _get_nlp_status(),
    }


@app.get("/health/nlp")
async def nlp_health_check():
    """Detailed NLP model loading status."""
    return _get_nlp_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

