import onnxruntime as rt
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import time

model = 'mobilenetv2.onnx'
sess = rt.InferenceSession(model)
input_name = sess.get_inputs()[0].name

img = image.load_img('data/1.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
prediction = sess.run(None, {input_name: x})[0]

path = 'data/'
mes = []
n = 6 # batch size
for k in range(20):
    for i in range(int(12/n)):
        inp = np.ndarray((0,224,224,3), dtype=np.float32)
        for j in range(n):
            img = image.load_img(path + str(i*n+j) + '.jpg', target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            inp = np.append(inp, x, axis=0)
        t = time.time_ns()
        prediction = sess.run(None, {input_name: inp})[0]
        mes.append(time.time_ns() - t)
mes = np.array(mes)
avg = np.mean(mes, axis=0)
std = np.std(mes, axis=0)
print(avg / 10 ** 9)
print(std / 10 ** 9)

