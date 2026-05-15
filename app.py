import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from backend.analyzer import analyze_text

# Point Flask's static folder to frontend/ so it serves
# index.html, styles.css, script.js from there.
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

@app.route('/')
def serve_frontend():
    """Serve the main dashboard HTML file."""
    return send_from_directory('frontend', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # 1. Receive the data from the frontend
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = data['text']
    print(f"Received input for analysis: {input_text[:50]}...")

    # 2. Run the AI analysis pipeline.
    response_data = analyze_text(input_text)

    return jsonify(response_data)

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)
