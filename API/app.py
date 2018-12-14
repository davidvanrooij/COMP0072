
# encoding=utf8
import base64

from io import BytesIO
from PIL import Image
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from classify import ClassifyImage


APP = Flask(__name__)

# Allow CORS for all domains on all routes
CORS(APP)
APP.config['CORS_HEADERS'] = 'Content-Type'

@APP.route('/status')
@cross_origin()
def status():
    """Endpoint to check if API is live"""
    return "OK", 200


@APP.route('/image', methods=['POST'])
@cross_origin()
def image():
    """Classify numbers on POST request to /images"""

    # Required field is missing
    if 'imgBase64' not in request.form:
        return jsonify(error=400, text='Missing imgBase64 input field'), 400

    try:
        image_base = request.form['imgBase64']
        image_base = image_base.replace('data:image/png;base64,', '')
        image_decoded = base64.b64decode(image_base)
        image_array = np.array(Image.open(BytesIO(image_decoded)))[:, :, 3]

        try:
            classify = ClassifyImage()
            classify.set_img(image_array)
            return jsonify(classify.classify()), 200

        # Could not classify image
        except Exception as error:
            return jsonify(error=500, text=str(error)), 500

    # Base64 string contains an error or not an image
    except Exception as error:
        return jsonify(error=400, text='Cannot process imgBase64 string provided'), 400

@APP.errorhandler(404)
def not_found(e):
    """Returns 404 not found error's in a json format"""
    return jsonify(error=404, text=str(e)), 404

@APP.errorhandler(405)
def invalid_method(e):
    """Returns 405 invalid method error's in a json format"""
    return jsonify(error=405, text=str(e)), 405

if __name__ == "__main__":
    APP.run(host='127.0.0.1', port=5000)
