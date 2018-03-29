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
path = "data/"

images = []
sor = []
outputs = []

for i in range(0,400):
    io = ImageIO(path+"input_"+str(i)+".raw")
    img = io.read(size[0], size[1], size[2])
    images.append(img.get_whole_array())

for i in range(0,400):
    io = ImageIO(path+"output_"+str(i)+".raw")
    img = io.read(size[0], size[1], size[2])
    outputs.append(img.get_whole_array())

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

ksize = 88
isize = 152//2
jsize = 152//2

# on va travailler sur des morceaux de 88,
X = tf.placeholder(tf.float32, shape=[None, ksize,jsize,isize,1])
Y = tf.placeholder(tf.float32, shape=[None, ksize,jsize,isize,1])
training = tf.placeholder(tf.bool)



def cnn_model(x_train_data, keep_rate=0.7, seed=None):
    with tf.name_scope("layer_a"):
        conv1 = tf.layers.conv3d(inputs=x_train_data, filters=64, kernel_size=[7, 7, 7], padding='same',
                                 activation=tf.nn.relu)
        conv1_ = tf.layers.max_pooling3d(conv1, 2,2,padding='SAME')

        conv2 = tf.layers.conv3d(inputs=conv1_,  filters=128, kernel_size=[7, 7, 7], padding='same', activation=tf.nn.relu)
        conv2_ = tf.layers.max_pooling3d(conv2, 2, 2, padding='SAME')

        with tf.device('/cpu:0'):
            conv3 = tf.layers.conv3d(inputs=conv2_, filters=2048, kernel_size=[5, 5, 5], padding='same', activation=tf.nn.relu)
            conv3_ = tf.layers.max_pooling3d(conv3, 2, 1, padding='SAME')

            with tf.name_scope("fully_con"):
                reshape = tf.reshape(conv3_, [-1, ksize, isize, jsize, 4])
                dense = tf.layers.dense(inputs=reshape, units=1024, activation=tf.nn.relu)
             # (1-keep_rate) is the probability that the node will be kept
                dropout = tf.layers.dropout(inputs=dense, rate=1-keep_rate, training=training)
            with tf.name_scope("y_conv"):
                 y_conv = tf.layers.dense(dropout, 1, activation=None)

    return y_conv

yprev = cnn_model(X)
loss_fonction = tf.losses.mean_squared_error(Y, yprev)
accurancy = 100*yprev/Y
adam_optimizer = tf.train.AdamOptimizer().minimize(loss_fonction)
batch_size = 1

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

load = 1





with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    if (load == 1):
        saver.restore(session, tf.train.latest_checkpoint("model_full/"))
        load = 0
    tf.get_default_graph().finalize()

    for train in range(10000) :

        for percent in range(100):
            rand = np.random.randint(320-batch_size)
            print("Random train = ", rand);
            data = np.zeros((batch_size, ksize, isize, jsize, 1))
            result = np.zeros((batch_size, ksize, isize, jsize, 1))

            img = images[rand]
            out = outputs[rand]
            imin = np.random.randint(175 - ksize)
            jmin = np.random.randint(152 - isize)
            kmin = np.random.randint(152 - jsize)

            for r in range(batch_size):
                for i in range(ksize):
                    for j in range(isize):
                        for k in range(jsize):
                            data[r, i, j, k, 0] = img[imin + i, jmin + j, k + kmin]
                            value = out[imin + i, jmin + j, k + kmin]
                            result[r, i, j, k, 0] = value / 4.

            loss, y = session.run([loss_fonction, yprev],
                                    feed_dict={X:data , Y: result, training: True})
            # Save the variables to disk.
            save_path = saver.save(session, "model_full/conv_full.ckpt", global_step=train*100+percent)

            if percent % 5 == 0:
                file = open("compute.log", 'a')
                log = "Train= " + str(train) + " Percent = " + str(percent) + " Loss = " + str(
                    loss)+"\n"

                file.writelines(log)
                file.close()
                print("Percent = ", percent, " Loss = ", loss);

        # Testing
        rand = np.random.randint(80)
        data = np.reshape(images[320+rand], [1, 175, 152, 152, 1])
        result = np.reshape(outputs[320+rand], [1, 175, 152, 152, 1])/4
        loss, y = session.run([loss_fonction, yprev],
                      feed_dict={X: data, Y: result, training: False})

        file = open("compute.log", 'a')
        log = "Test Data= " + str(rand) + " Loss = " + str(
               loss)+"\n"

        file.writelines(log)
        file.close()
