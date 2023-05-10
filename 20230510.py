#


# pip install grpcio
# pip install grpcio-tools
# pip install protobuf
#   After you install protobuf, which is like a compiler, you can use 【protoc --version】 to check the compiler version.
#   The following two commands are the same:
#   1) protoc --proto_path=protos/ --python_out=. --grpc_python_out=. protos/my_first.proto
#   2) python -m grpc_tools.protoc --proto_path=protos/ --python_out=. --grpc_python_out=. protos/my_first.proto
#   You can use either one above to do the thing : accelerate the serialization/deserialization process time compared to use JSON/XML format.
#   However, with way1, you need to build protoc environment in your PC. Do It Yourself.

# In this video, you should know why we need two files, one is this .py file, and the other one is a .proto file.
# Because grpc needs proto file to convert to binary code, we need the proto file.
# And if you use proto file, you also need its compiler, which is protobuf. That's why we need the addtional command.

# pip install tensorflow-serving-api
#   To deploy your Tensorflow model with api form, this package gives the total solution.
#   Whether you want to make a REST API or a gRPC communication protocol, it can quickly achieve it.

# docker run -p 8500:8500 --mount type=bind,source=$(pwd)/models,target=/models -e MODEL_NAME=mnist_model tensorflow/serving

import grpc
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2



# 建立gRPC通道
channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# 定義要進行預測的輸入數據
from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
input_data = np.array([x_test[111]])


model_spec = model_pb2.ModelSpec()
model_spec.name = 'mnist_model'
model_spec.version.value = 1
request = predict_pb2.PredictRequest(model_spec=model_spec)
request.inputs['flatten_input'].CopyFrom(tf.make_tensor_proto(input_data, dtype=np.float32))

# 發送請求
result = stub.Predict(request)

# 解析預測結果
output_data = tf.make_ndarray(result.outputs['dense_1']).astype(np.float32)
print(output_data)

import matplotlib.pyplot as plt
plt.imshow(x_test[111], cmap='gray')
plt.show()



# In this video, I'll show you how to use Tensorflow Serving as Server with gRPC api framework to predict MNIST data.
# I divide it into following steps:
#   1. Prepare your model (you must use this ~)
#   2. Prepare a proto file like this~
#   3. Enter the following commands in PS:
#       1) python -m grpc_tools.protoc --proto_path=protos/ --python_out=. --grpc_python_out=. protos/my_first.proto
#           This is for generating my_first_pb2_grpc.py & my_first_pb2.py files.
#           And about protobuf, you should know~
#       2) docker run -p 8500:8500 --mount type=bind,source=$(pwd)/models,target=/models -e MODEL_NAME=mnist_model tensorflow/serving
#           This is for building our tensorflow Server API with gRPC framework
#   Let's run all the steps again~ OK~ THX FOR WATCHING THIS VIDEO!!!