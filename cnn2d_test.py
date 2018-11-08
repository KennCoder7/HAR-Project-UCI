import tensorflow as tf
import numpy as np
from sklearn import metrics

test_x = np.load('UCI_data/processed/np_test_x.npy')
test_y = np.load('UCI_data/processed/np_test_y.npy')
print("### test_x (data) shape: ", test_x.shape, " ###")
print("### test_y (labels) shape: ", test_y.shape, " ###")

saver = tf.train.import_meta_graph("./model/HAR-UCI_model_1.meta")
with tf.Session() as session:
    saver.restore(session, tf.train.latest_checkpoint("./model/"))
    graph = tf.get_default_graph()
    feed_dict = {"X:0": test_x, "Y:0": test_y}
    acc = graph.get_tensor_by_name("accuracy:0")
    test_acc = session.run(acc, feed_dict=feed_dict)
    print("Test Accuracy:", test_acc)
    argmax_y_ = graph.get_tensor_by_name("argmax_y_:0")
    pred_y = session.run(argmax_y_, feed_dict={"X:0": test_x})
    # print(pred_y.shape)  # (2947,)
    cm = metrics.confusion_matrix(np.argmax(test_y, 1), pred_y, )
    print(cm)

# Test Accuracy: 0.90430945
# [[471   1  24   0   0   0]
#  [ 16 433  22   0   0   0]
#  [  2   1 417   0   0   0]
#  [  0   8   0 420  62   1]
#  [  1   3   0 118 410   0]
#  [  0  21   0   0   2 514]]


