import os
import tensorflow as tf
from ImageIO import ImageIO
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

size = [175, 152, 152]
path = "Data/"

poro = []
rocktype = []

for i in range(0,400):
    io = ImageIO(path+"Input_"+str(i)+".raw")
    img = io.read(size[0], size[1], size[2])
    poro.append(np.mean(img.get_whole_array()))


with open("outputs/sor.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';');
        count = 0
        for row in reader:
            if count != 0:
                rocktype.append(row[1])
            count +=1


fig, ax = plt.subplots()

ax.plot(rocktype, poro, 'g*')

plt.show()


