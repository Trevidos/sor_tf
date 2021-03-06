import os
import tensorflow as tf
from ImageIO import ImageIO
import numpy as np
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

size = [175, 152, 152]
path = "Data/"

images = []
sor = []

for i in range(0,400):
    io = ImageIO(path+"Input_"+str(i)+".raw")
    img = io.read(size[0], size[1], size[2])
    images.append(img.get_whole_array())


with open("outputs/sor.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';');
        count = 0
        for row in reader:
            if count != 0:
                sor.append(float(row[2]))
            count +=1

def extract(img, imin, imax, jmin, jmax, kmin, kmax):
    array = img.get_whole_array()
    subImage = array[imin:imax,jmin:jmax,kmin:kmax]
    return subImage

# on va travailler sur des morceaux de 22,19,19
X = tf.placeholder(tf.float32, shape=[None, 175,152,152,1])
Y = tf.placeholder(tf.float32, shape=[None,1])
training = tf.placeholder(tf.bool)



def cnn_model(x_train_data, keep_rate=0.7, seed=None):
    with tf.name_scope("layer_a"):
        conv1 = tf.layers.conv3d(inputs=x_train_data, filters=8, kernel_size=[7, 7, 7], padding='same',
                                 activation=tf.nn.relu)

        print(tf.shape(conv1))

        pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[5, 5, 5], strides=3)
        print(tf.shape(pool1))

        conv2 = tf.layers.conv3d(inputs=pool1,  filters=8, kernel_size=[5, 5, 5], padding='same', activation=tf.nn.relu)
        print(tf.shape(conv2))

        pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=3)

        conv3 = tf.layers.conv3d(inputs=pool2, filters=16, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu)
        print(tf.shape(conv3))

        pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[3, 3, 3], strides=2)
        print(tf.shape(pool3))

    with tf.name_scope("fully_con"):
        flattened = tf.reshape(pool3, [-1, 9*7*7*16])
        dense = tf.layers.dense(inputs=flattened, units=1024, activation=tf.nn.relu)
         # (1-keep_rate) is the probability that the node will be kept
        dropout = tf.layers.dropout(inputs=dense, rate=keep_rate, training=training)

    with tf.name_scope("y_conv"):
         y_conv = tf.layers.dense(dropout, 1, activation=None)

    return y_conv

yprev = cnn_model(X)
loss_fonction = tf.losses.mean_squared_error(Y, yprev)
accurancy = 100*yprev/Y
adam_optimizer = tf.train.AdamOptimizer().minimize(loss_fonction)
batch_size = 5

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

load = 1

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter('./logs/', session.graph)
    if (load == 1):
        saver.restore(session, "models/convol_sor.ckpt")
        load = 0

    for train in range(100) :

        rand = np.random.randint(320-batch_size)
        print("Random train = ", rand);
        data = np.reshape(images[rand: rand +batch_size], [batch_size, 175, 152, 152, 1])
        result = np.reshape(sor[rand: rand+batch_size], [batch_size, 1])

        for i in range(100):
            loss, y = session.run([loss_fonction, yprev],
                                    feed_dict={X:data , Y: result, training: True})


            if i%5==0:
                print("Percent = ",i, " Loss = ", loss);

        # Save the variables to disk.
        save_path = saver.save(session, "models/convol_sor.ckpt")


        # Testing
        rand = np.random.randint(80)
        data = np.reshape(images[320+rand], [1, 175, 152, 152, 1])
        result = np.reshape(sor[320+rand], [1, 1])
        loss, y = session.run([loss_fonction, yprev],
                    feed_dict={X: data, Y: result, training: False})

        print("On Testing Data(",rand,"): ", loss)