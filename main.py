from loguru import logger
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from src.main_router import main_router
from src.ml_queue_integration import initialize_ml_queue
from config import settings


app = FastAPI(
    title='DLS detection',
    description='',
    version='0.0.1',
    root_path="/api",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Инициализация ML очереди при запуске приложения"""
    try:
        initialize_ml_queue()
        logger.info("ML queue initialized on startup")
    except Exception as e:
        logger.error(f"Failed to initialize ML queue: {e}")


app.include_router(main_router)
