from flask import Flask, jsonify
import time # For prototyping
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/data', methods=['GET'])
def get_data():
    time.sleep(30)  # Delay response by 30 seconds
    response_data = {
        "leak": True,
        "distance": 10
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
