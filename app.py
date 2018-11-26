from classify import *

from flask import Flask, request, jsonify

import base64
from PIL import Image
from io import BytesIO
import numpy as np
from json import dumps


app = Flask(__name__)

@app.route('/status')
def status():
    return "OK", 200
 

@app.route('/image', methods=['POST'])
def image():

	image_base = request.form['imgBase64']

	image_base = image_base.replace('data:image/png;base64,', '')
	im = np.array(Image.open(BytesIO(base64.b64decode(image_base))))[:, :, 3]

	c = ClassifyImage()
	c.set_img(im)

	return dumps(c.classify()), 200


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)