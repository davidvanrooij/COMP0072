
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

    if 'imgBase64' not in request.form:
        return 'Missing imgBas64 input field', 400

    try:
        image_base = request.form['imgBase64']

        image_base = image_base.replace('data:image/png;base64,', '')
        image_array = np.array(Image.open(BytesIO(base64.b64decode(image_base))))[:, :, 3]

        classify = ClassifyImage()
        classify.set_img(image_array)

        return jsonify(classify.classify()), 200

    except Exception as error:
        print(error)
        return 'Something went wrong: ' + str(error), 500


if __name__ == "__main__":
    APP.run(host='127.0.0.1', port=5000)
