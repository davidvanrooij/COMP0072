from classify import *

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import base64
from PIL import Image
from io import BytesIO
import numpy as np
from json import dumps


app = Flask(__name__)

# Allow CORS for all domains on all routes
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/status')
@cross_origin()
def status():
    return "OK", 200
 

@app.route('/image', methods=['POST'])
@cross_origin()
def image():

	if not('imgBase64' in request.form):
		return 'Missing imgBas64 input field', 400

	try:
		image_base = request.form['imgBase64']

		image_base = image_base.replace('data:image/png;base64,', '')
		im = np.array(Image.open(BytesIO(base64.b64decode(image_base))))[:, :, 3]

		c = ClassifyImage()
		c.set_img(im)

		return dumps(c.classify()), 200

	except Exception as e:
		print(e);
		return 'Something went wrong: ' + str(e), 500


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)