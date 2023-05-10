import tensorflow as tf
from keras.datasets import mnist

# 載入數據集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 預處理數據
x_train = x_train / 255.0
x_test = x_test / 255.0

# 定義模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, epochs=5)

# 評估模型
model.evaluate(x_test, y_test)

import numpy as np
input_data = np.array([x_test[0]])
predictions = model.predict(input_data)

# 保存模型
model.save('mnist_model')

# We move the folder. And new a folder name '1' for recognizing the first version in our model.