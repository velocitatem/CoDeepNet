from flask import Flask
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, origins="*")

@app.route('/health')
def health():
    return {'status': 'healthy'}, 200


if __name__ == '__main__':
    PORT=int(os.getenv("BACKEND_PORT", 5000))
    app.run(host='0.0.0.0', port=PORT, debug=True)
