from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import os
from datetime import datetime

from dos.core.analyzer import DependencyGraphAnalyzer, ImpactAnalyzer, TemporalAnalyzer
from dos.ingestion.data_ingestor import DataIngestionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dependency Ops Sentinel (DOS) API",
    description="API for Dependency Ops Sentinel - An intelligent dependency monitoring and incident response system.",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
analyzer = DependencyGraphAnalyzer(input_dim=10)  # Example input dimension
impact_analyzer = ImpactAnalyzer()
temporal_analyzer = TemporalAnalyzer(input_dim=5)  # Example input dimension

# Initialize data ingestion manager
config = {
    'kubernetes': {
        'enabled': True,
        'namespace': os.getenv('K8S_NAMESPACE', 'default')
    },
    'cloud_metrics': {
        'enabled': True,
        'provider': os.getenv('CLOUD_PROVIDER', 'aws'),
        'regions': os.getenv('CLOUD_REGIONS', 'us-east-1').split(',')
    }
}

data_manager = DataIngestionManager(config)

# Models
class HealthCheckResponse(BaseModel):
    status: str
    version: str
    timestamp: str

class AnalysisRequest(BaseModel):
    component_id: str
    metrics: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    component_id: str
    anomaly_score: float
    root_cause_confidence: float
    severity: str
    recommendations: List[str]
    timestamp: str

# Routes
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_component(request: AnalysisRequest):
    """
    Analyze a component's metrics and return anomaly detection results.
    """
    try:
        # In a real implementation, we would process the metrics with our models
        # For now, we'll return mock responses
        
        # Mock analysis (replace with actual model inference)
        anomaly_score = 0.85  # Example score from 0-1
        root_cause_confidence = 0.75  # Example confidence score
        
        # Get impact analysis
        impact = impact_analyzer.analyze_impact(
            anomaly_scores=[anomaly_score],
            root_cause_probs=[root_cause_confidence],
            metadata=request.metadata or {}
        )
        
        return {
            "component_id": request.component_id,
            "anomaly_score": anomaly_score,
            "root_cause_confidence": root_cause_confidence,
            "severity": impact['severity'],
            "recommendations": impact['recommendations'],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collect-metrics")
async def collect_metrics():
    """
    Collect metrics from all configured data sources.
    """
    try:
        metrics = await data_manager.collect_data()
        return {"status": "success", "data": metrics}
    except Exception as e:
        logger.error(f"Error collecting metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "dos.api.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development"
    )
