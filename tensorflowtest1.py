# import tensorflow_hub as hub
import cv2
# import numpy
# import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt

print(tf.__version__)

width = 1028
height = 1028

# Load image by Opencv2
img = cv2.imread(r'G:\study\python\learn\tensorflow\image_2.jpg')
# Resize to respect the input_shape
inp = cv2.resize(img, (width, height))

# Convert img to RGB
rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

# COnverting to uint8
rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

# Add dims to rgb_tensor
rgb_tensor = tf.expand_dims(rgb_tensor, 0)

height, width, channels = img.shape
print(height, width, channels)

cv2.imshow("image", inp)
cv2.waitKey(100000)
