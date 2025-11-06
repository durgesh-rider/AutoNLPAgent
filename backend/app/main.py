from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers.upload import router as upload_router

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A no-code web-based autonomous NLP platform",
    version=settings.version,
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include routers
app.include_router(upload_router)
from app.routers.tasks import router as tasks_router
app.include_router(tasks_router)
from app.routers.training import router as training_router
app.include_router(training_router)
from app.routers.evaluation import router as evaluation_router
app.include_router(evaluation_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.version}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to AutoNLP-Agent",
        "version": settings.version,
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )