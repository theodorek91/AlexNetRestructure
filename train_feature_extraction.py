import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time
from scipy.misc import imread
import pandas as pd

OutputFile = open("C:\\Users\\tking\\CarND-Alexnet-Feature-Extraction\\OutputBatch1.txt","w")
OutputFile.write("test1")
nb_classes = 43
epochs = 5
batch_size = 100
sign_names = pd.read_csv('signnames.csv')

# TODO: Load traffic signs data.
with open('./train.p', 'rb') as f:
    train = pickle.load(f)
#train = pickle.load(open("train.p","rb"))


# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(train['features'], train['labels'], test_size=0.33, random_state=0)


# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None,32,32,3))
labels = tf.placeholder(tf.int64,None)
resized = tf.image.resize_images(features,(227,227))


# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)


# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits )
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list=[fc8W, fc8b])

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
init_op = tf.global_variables_initializer()

# TODO: Train and evaluate the feature extraction model.
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_op, feed_dict={features: batch_x, labels: batch_y})
        loss = sess.run(loss_op, feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_accuracy / num_examples, total_loss / num_examples
print('done')

# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(train_op, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})
        val_acc, val_loss = evaluate(X_valid, y_valid)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")
        OutputFile.write("Epoch" +str(i+1))
        OutputFile.write("Time: %.3f seconds" % (time.time() - t0))
        OutputFile.write("Validation Loss =" + str(val_loss))
        OutputFile.write("Validation Accuracy =" +str(val_acc))
        OutputFile.write("")

    output = sess.run(probs, feed_dict={features: [im1, im2]})
    tf.train.saver.save(sess,'my_model')
    #print("Image", input_im_ind)
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    OutputFile.write("Image" +str(input_im_ind))
    for i in range(5):
        print("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
        OutputFile.write("%s: %.3f" % (sign_names.ix[inds[-1 - i]][1], output[input_im_ind, inds[-1 - i]]))
    print()
OutputFile.close()
