import tensorflow as tf
import numpy as np
from sklearn import metrics

print("### Process1 --- data load ###")
data = np.load('UCI_data/processed/np_train_x.npy')
labels = np.load('UCI_data/processed/np_train_y.npy')
test_x = np.load('UCI_data/processed/np_test_x.npy')
test_y = np.load('UCI_data/processed/np_test_y.npy')
print("### labels shape: ", labels.shape, " ###")
print("### Process2 --- data spilt ###")
train_test_split = np.random.rand(len(data)) < 0.70
train_x = data[train_test_split]
train_y = labels[train_test_split]
valid_x = data[~train_test_split]
valid_y = labels[~train_test_split]
print("### train_x (data) shape: ", train_x.shape, " ###")
print("### train_y (labels) shape: ", train_y.shape, " ###")
print("### valid_x (data) shape: ", valid_x.shape, " ###")
print("### valid_y (labels) shape: ", valid_y.shape, " ###")

# define
seg_height = 36
seg_len = 128
num_channels = 1
num_labels = 6
batch_size = 100
learning_rate = 0.001
num_epoches = 100
num_batches = train_x.shape[0] // batch_size
min_acc = -np.infty
print("### num_batch: ", num_batches, " ###")

training = tf.placeholder_with_default(False, shape=())
X = tf.placeholder(tf.float32, (None, seg_height, seg_len, num_channels), name='X')
Y = tf.placeholder(tf.float32, (None, num_labels), name='Y')

print("### Process3 --- define ###")


# convolution layer 1
conv1 = tf.layers.conv2d(
    inputs=X,
    filters=10,
    kernel_size=[5, 5],
    strides=[1, 1],
    padding='valid',
    activation=tf.nn.relu,
    name="conv1"
)
print("### convolution layer 1 shape: ", conv1.shape, " ###")

# pooling layer 1
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[4, 4],
    strides=[4, 4],
    padding='same',
    name="pool1"
)
print("### pooling layer 1 shape: ", pool1.shape, " ###")

# convolution layer 2
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=100,
    kernel_size=[5, 5],
    strides=[1, 1],
    padding='valid',
    activation=tf.nn.relu,
    name="conv2"
)
print("### convolution layer 2 shape: ", conv2.shape, " ###")

# pooling layer 2
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2, 4],
    strides=[2, 4],
    padding='valid',
    name="pool2"
)
print("### pooling layer 2 shape: ", pool2.shape, " ###")

shape = pool2.get_shape().as_list()
flat = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

# fully connected layer 1
fc1 = tf.layers.dense(
    inputs=flat,
    units=120,
    activation=tf.nn.relu,
    name="fc1"
)
# fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
print("### fully connected layer 1 shape: ", fc1.shape, " ###")

# softmax layer
sof = tf.layers.dense(
    inputs=fc1,
    units=num_labels,
    activation=tf.nn.softmax,
    name="softmax"
)
print("### softmax layer  shape: ", sof.shape, " ###")

y_ = sof
print("### prediction shape: ", y_.get_shape(), " ###")

saver = tf.train.Saver()
tf.add_to_collection("y_", y_)

loss = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0)), name="loss")


train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# with tf.name_scope("evaluation"):
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epoches):
        # cost_history = np.empty(shape=[0], dtype=float)
        for b in range(num_batches):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size)]
            batch_y = train_y[offset:(offset + batch_size)]
            _, c = session.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y})
            # cost_history = np.append(cost_history, c)
        if (epoch + 1) % 10 == 0:
            print("Epoch: ", epoch + 1, " Training Loss: ", c,
                  " Training Accuracy: ", session.run(accuracy, feed_dict={X: train_x, Y: train_y}))
        if (epoch + 1) % 50 == 0:
            print("Epoch: ", epoch + 1, "Valid Accuracy:", session.run(accuracy, feed_dict={X: valid_x, Y: valid_y}))
        if (epoch + 1) % 100 == 0:
            test_acc = session.run(accuracy, feed_dict={X: test_x, Y: test_y})
            print("Epoch: ", epoch + 1, "Test Accuracy:", test_acc)
            pred_y = session.run(tf.argmax(y_, 1), feed_dict={X: test_x})
            cm = metrics.confusion_matrix(np.argmax(test_y, 1), pred_y, )
            print(cm)
            n = 0
            if test_acc > min_acc:
                n += 1
                tf.train.Saver().save(session, "./model/HAR-UCI_model_" + str(n))
                print("### Save model_" + str(n) + " successfully ###")
                min_acc = test_acc

# 2018/11/5 10c5*5-p4*4-100c5*5-p2*4-fc120-sof AdamOptimizer
# Epoch:  100  Training Loss:  5.863849  Training Accuracy:  0.9759369
# Epoch:  100 Valid Accuracy: 0.9623138
# Epoch:  100 Test Accuracy: 0.93043774
# ### Save model_1 successfully ###
