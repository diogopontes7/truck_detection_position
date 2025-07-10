#!/usr/bin/env python3
"""
Truck Alignment API - FastAPI Application
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import yaml
import os
from pathlib import Path
from typing import List, Dict, Optional
import tempfile
import shutil

# Import your truck alignment checker
from truck_detection_position.app.truck_alignment_checker import TruckAlignmentChecker
from enhanced_checker import EnhancedTruckChecker

# Load configuration
def load_config():
    """Load configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Default configuration
        return {
            "alignment": {
                "vertical_tolerance": 0.30,
                "behind_tolerance": 0.80
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }

config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Truck Alignment API",
    description="API for checking truck component alignment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize checkers
checker = TruckAlignmentChecker()
enhanced_checker = EnhancedTruckChecker("datasetv2")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Truck Alignment API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "check_single": "/check-single",
            "check_batch": "/check-batch",
            "analyze_dataset": "/analyze-dataset",
            "config": "/config"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "truck-alignment-api"}

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "alignment": config["alignment"],
        "api": config["api"]
    }

@app.post("/check-single")
async def check_single_truck(
    label_data: str,
    vertical_tolerance: Optional[float] = None,
    behind_tolerance: Optional[float] = None
):
    """
    Check alignment for a single truck using label data
    
    Args:
        label_data: YOLO format label data as string
        vertical_tolerance: Optional override for vertical tolerance
        behind_tolerance: Optional override for behind tolerance
    """
    try:
        # Create temporary file for label data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(label_data)
            temp_label_path = f.name
        
        # Parse labels
        labels = checker.parse_label_file(temp_label_path)
        
        if not labels:
            raise HTTPException(status_code=400, detail="No valid labels found in data")
        
        # Override tolerances if provided
        if vertical_tolerance is not None:
            checker.vertical_tolerance = vertical_tolerance
        if behind_tolerance is not None:
            checker.behind_tolerance = behind_tolerance
        
        # Check truck position
        result = checker.check_truck_position(labels)
        
        # Clean up temporary file
        os.unlink(temp_label_path)
        
        return {
            "status": "success",
            "result": result,
            "labels_found": [l['class_name'] for l in labels],
            "tolerances_used": {
                "vertical": checker.vertical_tolerance,
                "behind": checker.behind_tolerance
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/check-batch")
async def check_batch_trucks(
    label_files: List[UploadFile] = File(...),
    vertical_tolerance: Optional[float] = None,
    behind_tolerance: Optional[float] = None
):
    """
    Check alignment for multiple trucks using uploaded label files
    """
    try:
        results = []
        
        for file in label_files:
            if not file.filename.endswith('.txt'):
                continue
                
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
                shutil.copyfileobj(file.file, f)
                temp_path = f.name
            
            # Parse and check
            labels = checker.parse_label_file(temp_path)
            
            if labels:
                # Override tolerances if provided
                if vertical_tolerance is not None:
                    checker.vertical_tolerance = vertical_tolerance
                if behind_tolerance is not None:
                    checker.behind_tolerance = behind_tolerance
                
                result = checker.check_truck_position(labels)
                results.append({
                    "filename": file.filename,
                    "result": result,
                    "labels_found": [l['class_name'] for l in labels]
                })
            
            # Clean up
            os.unlink(temp_path)
        
        return {
            "status": "success",
            "total_files": len(label_files),
            "processed_files": len(results),
            "results": results,
            "tolerances_used": {
                "vertical": checker.vertical_tolerance,
                "behind": checker.behind_tolerance
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.post("/analyze-dataset")
async def analyze_dataset(
    split: str = "test",
    export_results: bool = False,
    vertical_tolerance: Optional[float] = None,
    behind_tolerance: Optional[float] = None
):
    """
    Analyze entire dataset split (test/valid/train)
    """
    try:
        # Override tolerances if provided
        if vertical_tolerance is not None:
            enhanced_checker.vertical_tolerance = vertical_tolerance
        if behind_tolerance is not None:
            enhanced_checker.behind_tolerance = behind_tolerance
        
        # Analyze dataset
        results, summary = enhanced_checker.analyze_dataset(
            split=split, 
            export_results=export_results
        )
        
        return {
            "status": "success",
            "split": split,
            "summary": summary,
            "total_images": len(results),
            "tolerances_used": {
                "vertical": enhanced_checker.vertical_tolerance,
                "behind": enhanced_checker.behind_tolerance
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")

@app.post("/update-config")
async def update_config(
    vertical_tolerance: Optional[float] = None,
    behind_tolerance: Optional[float] = None
):
    """
    Update configuration parameters
    """
    try:
        if vertical_tolerance is not None:
            config["alignment"]["vertical_tolerance"] = vertical_tolerance
            checker.vertical_tolerance = vertical_tolerance
            enhanced_checker.vertical_tolerance = vertical_tolerance
            
        if behind_tolerance is not None:
            config["alignment"]["behind_tolerance"] = behind_tolerance
            checker.behind_tolerance = behind_tolerance
            enhanced_checker.behind_tolerance = behind_tolerance
        
        # Save updated config
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)
        
        return {
            "status": "success",
            "message": "Configuration updated",
            "new_config": config["alignment"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=True
    ) 