import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# Point Flask's static folder to the project root so it serves
# index.html, styles.css, script.js directly.
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def serve_frontend():
    """Serve the main dashboard HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    # 1. Receive the data from the frontend
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    input_text = data['text']
    print(f"Received input for analysis: {input_text[:50]}...")

    # 2. Simulate AI processing time (e.g., waiting for LLMs to debate)
    # The frontend animation takes about 10 seconds, but we can return early
    # or sleep here to simulate heavy workload if needed.
    time.sleep(2) 

    # 3. In a real app, this is where your AI model would run.
    # For now, we return mock dynamic data based on the input length as a basic proof of concept.
    
    if len(input_text) < 50:
        verdict = "INCONCLUSIVE"
        color_class = "inconclusive" # We will map this in CSS/JS
        confidence = "45.0%"
        justification_details = "The provided text is too short to accurately assess its veracity using the multi-agent debate framework."
        key_facts = ["Text length is insufficient for contextual analysis."]
        links = []
    else:
        # Mocking the Manipulated Content response
        verdict = "MANIPULATED CONTENT"
        color_class = "manipulated" 
        confidence = "98.4%"
        justification_details = "The text you provided was processed by the Python backend. This demonstrates that the LLM engine successfully received the input and returned a verified mock response."
        key_facts = [
            "Input Received: The Python Flask backend successfully intercepted the POST request.",
            "Processing Simulation: The backend simulated an analysis delay.",
            "Dynamic Output: This JSON payload was constructed dynamically and sent back to the HTML interface."
        ]
        links = [
            {
                "source": "Python Backend",
                "title": f"Analyzed input starting with: '{input_text[:20]}...'"
            },
            {
                "source": "System Setup",
                "title": "Flask and Flask-CORS successfully initialized and responding."
            }
        ]

    # 4. Construct the JSON response
    response_data = {
        "verdict": verdict,
        "color_class": color_class,
        "confidence": confidence,
        "justification": justification_details,
        "key_facts": key_facts,
        "links": links
    }

    return jsonify(response_data)

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)
