"""
FastAPI Application — K8s Cluster Monitoring
=============================================
Real-time Kubernetes anomaly detection dashboard and API.
"""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from app.routes.alerts   import router as alerts_router
from app.routes.health   import router as health_router
from app.routes.metrics  import router as metrics_router
from app.routes.pipeline import router as pipeline_router
from mlops_poc.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("K8s Monitoring API starting up...")
    model_path = __import__("pathlib").Path("artifacts/model_trainer/anomaly_detector.joblib")
    if model_path.exists():
        from mlops_poc.pipeline.prediction_pipeline import PredictionPipeline
        app.state.pipeline = PredictionPipeline()
        _ = app.state.pipeline.model  # pre-warm
        logger.info("Anomaly detector pre-warmed ✓")
    else:
        logger.warning("Model not found — run: python main.py")
    yield
    logger.info("K8s Monitoring API shutting down...")


app = FastAPI(
    title="K8s Cluster Monitoring — MLOps POC",
    description="Real-time ML-powered Kubernetes anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    logger.info("%s %s → %d (%.1fms)", request.method, request.url.path, response.status_code, ms)
    return response


app.include_router(health_router)
app.include_router(alerts_router)
app.include_router(metrics_router)
app.include_router(pipeline_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return FileResponse("templates/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8080)),
        reload=os.getenv("APP_ENV", "development") == "development",
    )
