import numpy as np
import jsonpickle
import cv2
import os

from flask import Flask, request, Response, jsonify
from datetime import datetime
from mmdeploy_runtime import Detector

app = Flask(__name__)
classes = ('ayam', 'blueberry-muffin', 'bubur', 'burger',
           'chocolate-chip-cookie', 'croissant', 'doughnut',
           'es-pisang-ijo', 'french-fries', 'ikan-goreng', 'kacang-mete',
           'kangkung', 'klepon', 'kopi-hitam', 'macaroni-cheese',
           'martabak-manis', 'mie-ayam', 'nasi-goreng', 'nasi-putih',
           'onde-onde', 'pancake', 'pempek', 'pizza', 'red-velvet',
           'rendang', 'roti-slice', 'salad', 'salmon', 'sate',
           'sayur-asem', 'seafood', 'semangka', 'soto-ayam',
           'spaghetti-bolognese', 'steak', 'sushi-makizushi',
           'sushi-nigiri', 'telur-balado', 'telur-dadar',
           'telur-mata-sapi')


@app.route("/")
def test_connect():
    print("Echoing")
    response = {'status': 'success', 'message': 'Hello There!', 'detection': []}
    return jsonify(response)


@app.route("/upload", methods=['POST'])
def upload_img():
    # check if the post request has the file part
    start_time = datetime.now()
    
    if 'pic' not in request.files:
        print('No file part')
        response = {'status': 'fail', 'message': 'no file part'}
        return Response(response=jsonpickle.encode(response), status=400, mimetype='application/json')

    file = request.files['pic'].read()
    nparr = np.fromstring(file, np.uint8)

    response = dict(
        status="success",
        detection=[]
    )

    # decode img
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # create a detector
    detector = Detector(model_path='/home/dabestevanzzacc/model', device_name='cpu', device_id=0)
    # run the inference
    bboxes, labels, _ = detector(img)
    # Filter the result according to threshold
    indices = [i for i in range(len(bboxes))]
    for index, bbox, label_id in zip(indices, bboxes, labels):
        [left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
        if score < 0.5:
            continue
        data = dict(
            left=float(left),
            top=float(top),
            right=float(right),
            bottom=float(bottom),
            label=classes[label_id],
            score=round(float(score), 3)
        )
        response['detection'].append(data)
        
    dt = datetime.now() - start_time
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    print("####################")
    print(f"Time Taken: {int(ms)}ms")
    print("####################")
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT", default=5000))
