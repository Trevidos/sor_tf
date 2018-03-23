import os
import tensorflow as tf
from ImageIO import ImageIO
import numpy as np
import csv
import unpool_3d
import math

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


size = [175, 152, 152]
path = "data/"

images = []
sor = []

batch_size = 20

outputs = []
for i in range(0,400):
    io = ImageIO(path+"input_"+str(i)+".raw")
    img = io.read(size[0], size[1], size[2])
    images.append(img.get_whole_array())

for i in range(0,400):
    io = ImageIO(path+"output_"+str(i)+".raw")
    img = io.read(size[0], size[1], size[2])
    outputs.append(img.get_whole_array())


with open("outputs/sor.csv", 'rU') as csvfile:
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

ksize = 24
isize = 24
jsize = 24

# on va travailler sur des morceaux de 25,22,22
X = tf.placeholder(tf.float32, shape=[None, ksize, isize, jsize,1])
Y = tf.placeholder(tf.float32, shape=[None, ksize, isize, jsize,1])
training = tf.placeholder(tf.bool)

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat(axis=i, values=[out, tf.zeros_like(out)])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

def pool(value, name='pool'):
    """Downsampling operation.
    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, d0/2, d1/2, ..., dn/2, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        out = value
        for sh_i in sh[1:-1]:
            assert sh_i % 2 == 0
        for i in range(len(sh[1:-1])):
            out = tf.reshape(out, (-1, 2, np.prod(sh[i + 2:])))
            out = out[:, 0, :]
        out_size = [-1] + [int(math.ceil(s / 2)) for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def dense_model(x_train_data, keep_rate=0.8, seed=None):
    dense = tf.layers.dense(x_train_data, units=256, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=keep_rate, training=training)
    y_conv = tf.layers.dense(dropout, units=256, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=y_conv, rate=keep_rate, training=training)
    final = tf.layers.dense(dropout2, 1, activation=tf.nn.softmax)
    return final

def cnn_model(x_train_data, keep_rate=0.8, seed=None):
    with tf.name_scope("layer_a"):
          conv1 = tf.layers.conv3d(inputs=x_train_data, filters=128, kernel_size=[3, 3, 3], padding='same',
                                   activation=tf.nn.relu, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        # # print(tf.shape(conv1))
        # #
          pool1 = pool(conv1)
        # # print(tf.shape(pool1))
        #
          conv2 = tf.layers.conv3d(inputs=pool1,  filters=256, kernel_size=[2, 2, 2], padding='same', activation=tf.nn.relu, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        # # print(tf.shape(conv2))
        # #
          #pool2 = pool(conv2)
          # pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[3, 3, 3], strides=2)
        #
          #conv3 = tf.layers.conv3d(inputs=pool2, filters=128, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
          #pool3 = pool(conv3)
          # pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[2, 2, 2], strides=2)


    with tf.name_scope("fully_con"):

        # flattened = tf.reshape(x_train_data, [-1, 5*5*5*128])
        dense = tf.layers.dense(pool1, units=256, activation=tf.nn.relu)
        deconv1 = tf.layers.conv3d_transpose(dense, filters=128, kernel_size=[3, 3, 3], padding='same',
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01,
                                                                                                dtype=tf.float32))
        unpool1 = unpool(deconv1)
        deconv1 =tf.layers.conv3d_transpose(unpool1, filters=128, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        #unpool2 = unpool(deconv1)
        #deconv2=tf.layers.conv3d_transpose(unpool2, filters=256, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        #unpool3 = unpool(deconv2)
        #deconv3 = tf.layers.conv3d_transpose(unpool3, filters=128, kernel_size=[3, 3, 3], padding='same', activation=tf.nn.relu, kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
         # (1-keep_rate) is the probability that the node will be kept
        # dropout = tf.layers.dropout(inputs=dense, rate=keep_rate, training=training)
        y_conv = tf.layers.dense(deconv1, units=32, activation=tf.nn.relu)
        #dropout2 = tf.layers.dropout(inputs=y_conv, rate=keep_rate, training=training)
        final = tf.layers.dense(y_conv, 1, activation=None)


    # with tf.name_scope("deconv"):
    #     deconv1 = tf.layers.conv3d_transpose(dropout, 384, [3, 3, 3], padding='same')
    #     deconv2 = tf.layers.conv3d_transpose(deconv1, 256, [3, 3, 3], padding='same')

    # return tf.reshape(final, [batch_size*24*22*22,3])
    return final

yprev = cnn_model(X)
loss_fonction = tf.losses.mean_squared_error(Y, yprev)
adam_optimizer = tf.train.AdamOptimizer().minimize(loss_fonction)

correct_prediction = tf.logical_and(tf.greater(yprev, 0.5), tf.equal(Y, 1.0))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, 'float'))/tf.reduce_sum(tf.cast(tf.equal(Y, 1.0), 'float'))



# Add ops to save and restore all the variables.
saver = tf.train.Saver()

load = 0


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config=tf.ConfigProto(device_count={"GPU": 0, "CPU": 1})

with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())

    #file_writer = tf.summary.FileWriter('./logs/', session.graph)

    if (load == 1):
        saver.restore(session, "models/conv_small.ckpt")
        load = 0

    for train in range(10000) :

        rand = np.random.randint(320-batch_size)
        print("Random train = ", rand);
        data = np.zeros((batch_size, ksize, isize, jsize,1))
        result = np.zeros((batch_size, ksize, isize, jsize,1))

        img = images[rand]
        out = outputs[rand]

        for a in range(500):
            for r in range(batch_size):

                imin = np.random.randint(175-ksize)
                jmin =  np.random.randint(152-isize)
                kmin = np.random.randint(152 - jsize)

                for i in range(ksize):
                    for j in range(isize):
                        for k in range(jsize):
                            data[r, i, j, k, 0] = img[imin+i,jmin+j,k+kmin]
                            value = out[imin + i, jmin + j, k + kmin]

                            # if(value ==4):
                            #     value = 10

                            result[r, i, j, k, 0] = value / 4.
                            #     result[r,i,j,k,0] = 0
                            #     result[r, i, j, k, 1] = 1
                            # else :
                            #         result[r,i,j,k,0] = 1
                            #         result[r, i, j, k, 1] = 0



            adam, loss, yp,y, acc = session.run([adam_optimizer,loss_fonction, yprev, Y,accuracy],
                                   feed_dict={X:data , Y: result, training: True})

            if a%5==0:
                print("Percent = ",a/5, "Loss = ", loss, " Accuracy = ", acc, "Size of the Oil cluster = ", session.run(tf.reduce_sum(tf.cast(tf.greater(yp, 0.5), tf.float32))), " on =",
                              session.run(tf.reduce_sum(tf.cast(tf.equal(y, 1.0), tf.float32))));

        # Save the variables to disk.
        save_path = saver.save(session, "models/conv_small.ckpt")
