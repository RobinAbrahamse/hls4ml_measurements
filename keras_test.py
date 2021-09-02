import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import time

#loaded_model = tf.keras.applications.ResNet50V2()
loaded_model = tf.keras.applications.MobileNetV2()
img = image.load_img('data/1.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
prediction = loaded_model.predict(x)

path = 'data/'
mes = []
n = 6 # batch size
for k in range(20):
    for i in range(int(12/n)):
        inp = np.ndarray((0,224,224,3))
        for j in range(n):
            img = image.load_img(path + str(i*n+j) + '.jpg', target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            inp = np.append(inp, x, axis=0)
        t = time.time_ns()
        prediction = loaded_model.predict(inp)
        mes.append(time.time_ns() - t)
mes = np.array(mes)
avg = np.mean(mes, axis=0)
std = np.std(mes, axis=0)
print(avg / 10 ** 9)
print(std / 10 ** 9)

