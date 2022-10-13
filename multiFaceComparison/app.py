import base64
import json
import re
import time
from io import BytesIO

from flask import Flask, request
from werkzeug.utils import secure_filename

import compareFace
import compareFace2
import detectFace
import gunicorn

app = Flask(__name__)


def convert2Image(base64_str):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    return image_data


@app.route('/api/v1/compare2', methods=['POST'])
def compare_faces2():
    source_img_str = request.get_json()['source']
    target_img_str = request.get_json()['target']

    source_img_data = convert2Image(source_img_str)
    target_img_data = convert2Image(target_img_str)

    response = []
    start = time.time()
    distance, result = compareFace2.main(source_img_data, target_img_data)
    distance = float(distance)
    end = time.time()
    json_contect = {
        'result': str(result),
        'distance': round(distance, 2),
        'time_taken': round(end - start, 3),
    }
    response.append(json_contect)
    python2json = json.dumps(response)
    return app.response_class(python2json, content_type='application/json')


@app.route('/api/v1/compare', methods=['POST'])
def compare_faces():
    target = request.files['target']
    faces = request.files.getlist("faces")
    target_filename = secure_filename(target.filename)
    response = []
    for face in faces:
        start = time.time()
        distance, result = compareFace.main(target, face)
        distance = float(distance)
        end = time.time()
        json_contact = {
            'result': str(result),
            'distance': round(distance, 2),
            'time_taken': round(end - start, 3),
            'target': target_filename,
            'face': secure_filename(face.filename)
        }
        response.append(json_contact)
    python2json = json.dumps(response)
    return app.response_class(python2json, content_type='application/json')


@app.route('/api/v1/detect', methods=['POST'])
def detect_faces():
    faces = request.files.getlist("faces")
    # target_filename=secure_filename(target.filename)
    response = []
    for face in faces:
        start = time.time()
        _, result = detectFace.get_coordinates(face)
        end = time.time()
        json_contect = {
            'coordinates': result,
            'time_taken': round(end - start, 3),
            'image_name': secure_filename(face.filename)
        }
        response.append(json_contect)
    python2json = json.dumps(response)
    return app.response_class(python2json, content_type='application/json')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
