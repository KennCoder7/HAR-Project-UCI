import tensorflow as tf
import numpy as np

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

# Test Accuracy: 0.9321344


