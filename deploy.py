import tensorflow.compat.v1 as tf
from tensorflow.python import debug as tf_debug    #Adding to tensorboard debugger
tf.disable_v2_behavior()
import numpy as np
import sys
from layers import FcLayerDeploy, ConvLayerDeploy

def load_weights(directory, name):
	
	weights = np.load(directory + '/' + name + '-weights.npy')
	prune_mask = np.load(directory + '/' + name + '-prune-mask.npy')

	return weights, prune_mask

if __name__ == "__main__":
	
	weights_dir = './weights'
	
	x_PH = tf.placeholder(tf.float32, [None, 28, 28, 1])
	
	weights, prune_mask = load_weights(weights_dir, 'conv1')
	L1 = ConvLayerDeploy(weights, prune_mask, x_PH.shape[1], x_PH.shape[2], 2, 'conv1')
	x = L1.forward_matmul_preprocess(x_PH)
	x = tf.nn.relu(L1.forward_matmul(x))
	x = L1.forward_matmul_postprocess(x) # Output is 14x14x32
    #x = tf.nn.relu(L1.forward(x_PH))
	
	weights, prune_mask = load_weights(weights_dir, 'conv2')
	L2 = ConvLayerDeploy(weights, prune_mask, x.shape[1], x.shape[2], 2, 'conv2')
	x = L2.forward_matmul_preprocess(x)
	x = tf.nn.relu(L2.forward_matmul(x))
	x = L2.forward_matmul_postprocess(x)

	x = tf.reshape(x, [-1, 7 * 7 * 64])
	
	weights, prune_mask = load_weights(weights_dir, 'fc1')
	L3 = FcLayerDeploy(weights, prune_mask, 'fc1')	
	x = tf.nn.relu(L3.forward_matmul(x))
	
	weights, prune_mask = load_weights(weights_dir, 'fc2')
	L4 = FcLayerDeploy(weights, prune_mask, 'fc2')	
	logits = L4.forward_matmul(x)
		
	labels = tf.placeholder(tf.float32, [None, 10])
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	
	
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	
	sess = tf.Session()
	
	batches_acc = []
	for i in range(10):
	
		batch_x, batch_y = mnist.test.next_batch(1000)
		batch_x = np.reshape(batch_x,(-1, 28, 28, 1))	

		writer = tf.summary.FileWriter('./graphs', sess.graph)  # Added this line to enable tensorboard
		#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "cordelia:6067")  # adding debugger on cordelia PC in RRLAB on port 6006
		batch_acc = sess.run(accuracy,feed_dict={x_PH: batch_x, labels: batch_y})
		batches_acc.append(batch_acc)
				
	acc = np.mean(batches_acc)
		
	print('deploy accuracy:', acc)