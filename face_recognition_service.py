from flask import Flask, request, jsonify
import numpy as np
from typing import List
import time
from qdrant_client import QdrantClient
from configparser import ConfigParser

app = Flask(__name__)

class QdrantRecognition():
    def __init__(self, server_url, collection_name):
        self.client = QdrantClient(url=server_url)
        self.collection_name = collection_name

        # Check if the collection exists
        if collection_name not in [c.name for c in self.client.get_collections().collections]:
            raise ValueError(f"Collection {collection_name} not found in Qdrant server")
    
    def recognition(self, embeddings: List[np.ndarray]) -> tuple:
        """
        Perform recognition on a list of embeddings
        Args:
            embeddings: List of embeddings to recognize
        Returns:
            List of person names
            List of distances
        """
        person_names = []
        distances = []
        for embedding in embeddings:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                with_payload=True,
                limit=1
            ).points
            person_name = search_result[0].payload['person_name']   # person_name is stored in payload
            distance = search_result[0].score   # distance is stored in score
            app.logger.info(f"Person: {person_name}, Distance: {distance}")
            person_names.append(person_name)
            distances.append(float(distance))
        return person_names, distances

# Load configuration
config = ConfigParser()
config.read('config.ini')
QDRANT_URL = config.get('face_recognition', 'qdrant_server_url', fallback='http://localhost:6333')
COLLECTION_NAME = config.get('face_recognition', 'qdrant_collection_name', fallback='facesnet')

# Initialize recognition service
recognition_service = QdrantRecognition(QDRANT_URL, COLLECTION_NAME)

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """
    Endpoint to recognize faces from embeddings
    
    Expected JSON format:
    {
        "embeddings": [array of 128 float values]
    }
    
    Returns:
    {
        "names": [list of recognized names],
        "distances": [list of distances],
        "processing_time_ms": float
    }
    """    
    try:
        # Get data from request
        data = request.json
        
        if not data or 'embeddings' not in data:
            return jsonify({"error": "Missing embedding data"}), 400
        
        # Convert embedding to numpy array
        embeddings = data['embeddings']
        
        # Perform recognition
        names, distances = recognition_service.recognition(embeddings)
                
        # Return results
        return jsonify({
            "names": names,
            "distances": distances,
        })
    
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "service": "face-recognition-api"})

if __name__ == '__main__':
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
