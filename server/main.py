from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.utils.logger import Logger
from app.routers import video_processing

app = FastAPI(title="opencv-healthcare", version="1.0.0")
logger_instance = Logger()
logger = logger_instance.get_logger()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(video_processing.router)

@app.get("/")
async def root():
    logger.info("Root route accessed")
    return JSONResponse(content={"message": "Hello, World!"})

# Event handlers
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    # Perform startup tasks like connecting to databases

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application...")
    logger_instance.shutdown()
