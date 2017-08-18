from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

print(os.environ['PYTHONPATH'])

import cv2
import subprocess
import time
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from lib.datasets.factory import get_imdb
from tensorflow.python.saved_model.signature_constants import PREDICT_METHOD_NAME

print('starting server')
pid = subprocess.Popen(
    ["tensorflow_model_server", "--port=9000",
     "--model_name=frcnn", "--model_base_path=exported_models/"])

try:
    print("server starting")
    time.sleep(5)

    host, port = 'localhost', "9000"
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request

    imdb = get_imdb("lwir_humans_animals_1_train")
    num_images = len(imdb.image_index)
    #for i in range(num_images):
    for i in range(3):
        im = cv2.imread(imdb.image_path_at(i))

        # See prediction_service.proto for gRPC request/response details.
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'frcnn'
        request.model_spec.signature_name = "predict_post"
        request.inputs['image'].CopyFrom(
            tf.contrib.util.make_tensor_proto(im, shape=im.shape))

        result = stub.Predict(request, 10.0)  # 10 secs timeout
        print(result)
    #assert np.argmax(tensor_util.MakeNdarray(result.outputs["scores"])) == label[0]
finally:
    pid.kill()
    print("killed")