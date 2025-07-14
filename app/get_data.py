#!/usr/bin/env python3
"""
Simple API to receive truck alignment data
"""

from flask import Flask, request, jsonify
from datetime import datetime
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Store received data (in production, you'd use a database)
received_data = []

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Truck Alignment Data Receiver API",
        "version": "1.0.0",
        "endpoints": {
            "receive_data": "/webhook [POST]",
            "get_latest": "/latest [GET]",
            "get_all": "/data [GET]",
            "health": "/health [GET]",
            "stats": "/stats [GET]"
        },
        "status": "running"
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_records": len(received_data)
    })

@app.route('/webhook', methods=['POST'])
def receive_truck_data():
    """
    Main endpoint to receive truck alignment data
    This is where your inference.py will send data
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400
        
        # Add received timestamp
        data['received_at'] = datetime.now().isoformat()
        data['id'] = len(received_data) + 1
        
        # Store the data
        received_data.append(data)
        
        # Log the received data
        logger.info(f"Received truck data: {data.get('status')} - {data.get('message')}")
        
        # Print to console for debugging
        print(f"\nNEW TRUCK DATA RECEIVED:")
        print(f"   Status: {data.get('status')}")
        print(f"   Message: {data.get('message')}")
        print(f"   Image: {data.get('image_filename')}")
        print(f"   Timestamp: {data.get('timestamp')}")
        
        if 'details' in data:
            details = data['details']
            print(f"   Vertical Aligned: {details.get('vertically_aligned')}")
            print(f"   Behind Side: {details.get('behind_side_panel')}")
            print(f"   Distance: {details.get('distance_to_side')}")
                
        # Return success response
        return jsonify({
            "status": "success",
            "message": "Data received successfully",
            "id": data['id'],
            "received_at": data['received_at']
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            "error": "Error processing request",
            "details": str(e)
        }), 500

@app.route('/latest', methods=['GET'])
def get_latest_data():
    """Get the most recent truck data"""
    if not received_data:
        return jsonify({
            "message": "No data received yet"
        }), 404
    
    return jsonify(received_data[-1])

if __name__ == '__main__':
    print("🚀 Starting Truck Alignment Data Receiver API...")
    print("📡 Listening for truck alignment data...")
    print("🌐 Available endpoints:")
    print("   POST /webhook     - Receive truck data")
    print("   GET  /latest      - Get latest data")
    print("   GET  /data        - Get all data")
    print("   GET  /stats       - Get statistics")
    print("   GET  /health      - Health check")
    print("   POST /clear       - Clear all data")
    print("\n" + "="*50)
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=5000,       # Port 5000
        debug=True       # Enable debug mode
    )
