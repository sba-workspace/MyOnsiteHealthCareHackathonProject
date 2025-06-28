# test_fastapi_client.py - Example client usage

import requests
import json

# FastAPI server URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_prediction():
    """Test prediction endpoint"""
    # Sample coordinates
    test_data = {
        "coordinates": [
            {"lon": 73.17, "lat": 22.30},
            {"lon": 73.18, "lat": 22.31},
            {"lon": 73.19, "lat": 22.32},
            {"lon": 72.85, "lat": 21.17},
            {"lon": 72.86, "lat": 21.18},
            {"lon": 73.85, "lat": 18.52}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=test_data)
    print("\nPrediction Result:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_batch_prediction():
    """Test batch prediction endpoint"""
    # Simple coordinate arrays
    coordinates = [
        [73.17, 22.30],
        [73.18, 22.31], 
        [72.85, 21.17],
        [73.85, 18.52]
    ]
    
    response = requests.post(f"{BASE_URL}/predict-batch", json=coordinates)
    print("\nBatch Prediction Result:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_retrain():
    """Test model retraining"""
    training_data = {
        "training_coordinates": [
            [73.17, 22.30], [73.18, 22.31], [73.19, 22.32],  # Cluster 1
            [72.85, 21.17], [72.86, 21.18], [72.87, 21.19],  # Cluster 2  
            [73.85, 18.52], [73.86, 18.53], [73.87, 18.54],  # Cluster 3
            [74.12, 19.45], [74.13, 19.46]                   # Cluster 4
        ],
        "max_k": 10
    }
    
    response = requests.post(f"{BASE_URL}/retrain", json=training_data)
    print("\nRetrain Result:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/model-info")
    print("\nModel Info:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

if __name__ == "__main__":
    print("Testing FastAPI GeoClustering Service")
    print("=" * 50)
    
    # Test all endpoints
    try:
        # First check health
        health = test_health()
        
        if not health.get("model_loaded", False):
            print("\nModel not loaded, training first...")
            test_retrain()
        
        # Test predictions
        test_prediction()
        test_batch_prediction()
        test_model_info()
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to FastAPI server.")
        print("Make sure the server is running: python main.py")
    except Exception as e:
        print(f"Error: {e}")

# Node.js Integration Example (JavaScript)
nodejs_example = '''
// nodejs_fastapi_client.js
const axios = require('axios');

class GeoClusteringFastAPIService {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    async predictClusters(coordinates) {
        try {
            const response = await axios.post(`${this.baseUrl}/predict`, {
                coordinates: coordinates.map(coord => ({
                    lon: coord.lon || coord[0],
                    lat: coord.lat || coord[1]
                }))
            });
            
            return response.data;
        } catch (error) {
            throw new Error(`Prediction failed: ${error.response?.data?.error || error.message}`);
        }
    }

    async predictBatch(coordinates) {
        try {
            const response = await axios.post(`${this.baseUrl}/predict-batch`, coordinates);
            return response.data;
        } catch (error) {
            throw new Error(`Batch prediction failed: ${error.response?.data?.error || error.message}`);
        }
    }

    async retrainModel(trainingCoordinates, maxK = 15) {
        try {
            const response = await axios.post(`${this.baseUrl}/retrain`, {
                training_coordinates: trainingCoordinates,
                max_k: maxK
            });
            return response.data;
        } catch (error) {
            throw new Error(`Retraining failed: ${error.response?.data?.error || error.message}`);
        }
    }

    async getHealth() {
        try {
            const response = await axios.get(`${this.baseUrl}/health`);
            return response.data;
        } catch (error) {
            throw new Error(`Health check failed: ${error.message}`);
        }
    }

    async getModelInfo() {
        try {
            const response = await axios.get(`${this.baseUrl}/model-info`);
            return response.data;
        } catch (error) {
            throw new Error(`Model info failed: ${error.response?.data?.error || error.message}`);
        }
    }
}

// Express.js integration example
const express = require('express');
const app = express();
app.use(express.json());

const clusteringService = new GeoClusteringFastAPIService();

app.post('/cluster-coordinates', async (req, res) => {
    try {
        const { coordinates } = req.body;
        
        // Use the FastAPI service
        const result = await clusteringService.predictClusters(coordinates);
        
        // Process for your frontend
        const processedResult = {
            success: true,
            clusters: result.clusters.map(cluster => ({
                id: cluster.cluster_id,
                center: cluster.cluster_center,
                points: cluster.coordinates,
                count: cluster.coordinates.length
            })),
            summary: {
                total_points: result.total_points,
                total_clusters: result.total_clusters,
                processed_at: result.timestamp
            }
        };
        
        res.json(processedResult);
        
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

module.exports = GeoClusteringFastAPIService;
'''

print("\\n" + "="*50)
print("Node.js Integration Example:")
print("="*50)
print(nodejs_example)