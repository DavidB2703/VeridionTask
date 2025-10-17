from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os

# Add parent directory to path to import classifier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classifier import classify_company_emb_zs

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        company_text = data['text']

        if not company_text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Call the classification function
        predicted_labels = classify_company_emb_zs(company_text)

        return jsonify({
            'success': True,
            'predictions': predicted_labels
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("Loading models... This may take a moment.")
    print("Server will start once models are loaded.")
    app.run(debug=True, host='0.0.0.0', port=5000)

