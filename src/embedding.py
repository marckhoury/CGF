"""
    Usage

    Evaluation:
        python embedding.py --evaluate=True --checkpoint_model=<pretrained model file> --output_file=<output filename> --output_dimension=<output feature dimension> <input filename>
    
    Training:
        python embedding.py --evaluate=False --summaries_dir=<summaries dirname> --checkpoint_dir=<checkpoint dirname> --max_steps==<max iterations> <input filename>
"""
import sys, os
import time 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 910000, 'Number of steps to run trainer.') #previously 1551825
flags.DEFINE_integer('batch_size', 512, 'Number of examples in each minibatch.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_string('summaries_dir', './summaries/', 'Summaries directory.')
flags.DEFINE_string('checkpoint_dir', './checkpoints/', 'Checkpoint directory.')
flags.DEFINE_string('checkpoint_model', './checkpoints/embed_model_910000.ckpt', 'Pretrained model location.')
flags.DEFINE_float('margin', 1.0, 'Margin for contrastive loss.')
flags.DEFINE_boolean('evaluate', True, 'Load pretrained model.')
flags.DEFINE_string('output_file', 'features.npz', 'Output file for computed features.')
flags.DEFINE_integer('output_dimension', 32, 'Dimension of learned features. Make sure this value matches that of the pretrained model.')

def train(argv):
    npzfile = np.load(argv[1])
    data = npzfile['data']

    if not FLAGS.evaluate:
        triplets = npzfile['triplets']
    else:
        triplets = np.array([])
    print 'Data Loaded'
   
    if not FLAGS.evaluate:
        np.random.shuffle(triplets)

    dim = data.shape[1]
    num_examples = triplets.shape[0]
    
    train.index_in_epoch = 0
    train.epochs_completed = 0

    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x_a = tf.placeholder(tf.float32, [None, dim], name='x-anchor')
        x_p = tf.placeholder(tf.float32, [None, dim], name='x-positive')
        x_n = tf.placeholder(tf.float32, [None, dim], name='x-negative')
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(var, name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var-mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)
   
    def shared_fc_layer(input_tensor_anchor, input_tensor_positive, input_tensor_negative, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
            with tf.name_scope('Wx_plus_b_anchor'):
                preactivate_anchor = tf.matmul(input_tensor_anchor, weights) + biases
            with tf.name_scope('Wx_plus_b_positive'):
                preactivate_positive = tf.matmul(input_tensor_positive, weights) + biases
            with tf.name_scope('Wx_plus_b_negative'):
                preactivate_negative = tf.matmul(input_tensor_negative, weights) + biases

            activations_anchor = act(preactivate_anchor, 'activation_anchor')
            activations_positive = act(preactivate_positive, 'activation_positive')
            activations_negative = act(preactivate_negative, 'activation_negative')
            return (activations_anchor, activations_positive, activations_negative)

    def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
            activations = act(preactivate, 'activation')
            return activations

    with tf.name_scope('embedding'): 
        hidden1_anchor, hidden1_positive, hidden1_negative = shared_fc_layer(x_a, x_p, x_n, dim, 512, 'layer1')
        hidden2_anchor, hidden2_positive, hidden2_negative = shared_fc_layer(hidden1_anchor, hidden1_positive, hidden1_negative, 512, 512, 'layer2') 
        hidden3_anchor, hidden3_positive, hidden3_negative = shared_fc_layer(hidden2_anchor, hidden2_positive, hidden2_negative, 512, 512, 'layer3') 
        hidden4_anchor, hidden4_positive, hidden4_negative = shared_fc_layer(hidden3_anchor, hidden3_positive, hidden3_negative, 512, 512, 'layer4') 
        hidden5_anchor, hidden5_positive, hidden5_negative = shared_fc_layer(hidden4_anchor, hidden4_positive, hidden4_negative, 512, FLAGS.output_dimension, 'layer5', act=tf.identity)

    if not FLAGS.evaluate:
        with tf.name_scope('triplet_loss'):
            sub_pos = tf.subtract(hidden5_anchor, hidden5_positive)
            sub_neg = tf.subtract(hidden5_anchor, hidden5_negative)
            sq_pos = tf.square(sub_pos)
            sq_neg = tf.square(sub_neg)
            sq_dist_pos = tf.reduce_sum(sq_pos, 1)
            sq_dist_neg = tf.reduce_sum(sq_neg, 1)
            terms = tf.subtract(sq_dist_pos, sq_dist_neg) + FLAGS.margin
            relu = tf.nn.relu(terms)
            loss = tf.reduce_mean(relu)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

    tf.initialize_all_variables().run()

    saver = tf.train.Saver()

    def next_batch(batch_size):
        start = train.index_in_epoch
        train.index_in_epoch += batch_size
        if train.index_in_epoch > num_examples:
            train.epochs_completed += 1
            print 'Epochs completed = {0}'.format(train.epochs_completed) 
            start = 0
            train.index_in_epoch = batch_size
            assert batch_size < num_examples
        end = train.index_in_epoch
        
        anchor_indices, positive_indices, negative_indices = np.hsplit(triplets[start:end], [1,2])
        anchor_indices   = np.reshape(anchor_indices,   (anchor_indices.shape[0],))
        positive_indices = np.reshape(positive_indices, (positive_indices.shape[0],))
        negative_indices = np.reshape(negative_indices, (negative_indices.shape[0],))
       
        x_anchor = data[anchor_indices]
        x_positive = data[positive_indices]
        x_negative = data[negative_indices] 
        return x_anchor, x_positive, x_negative, triplets[start:end]

    def feed_dict():
        x_anchor, x_positive, x_negative, _ = next_batch(FLAGS.batch_size)
        return {x_a: x_anchor, x_p: x_positive, x_n: x_negative}
   
    if FLAGS.evaluate:
        saver.restore(sess, FLAGS.checkpoint_model)
        print 'Loaded saved model {0}.'.format(FLAGS.checkpoint_model)

        start = time.time()
        x_anchor = sess.run([hidden5_anchor], feed_dict={x_a: data})[0]
        end = time.time()
        print '{0} features computed in {1} seconds.'.format(len(data), end - start)
       
        np.savez_compressed(FLAGS.output_file, data=x_anchor)
        print 'Wrote file {0}.npz'.format(FLAGS.output_file)
    else: 
        iteration = 0
        losses = []
        accuracy = []
        for i in xrange(FLAGS.max_steps):
            if iteration % 1000 == 0:
                xa, xp, xn, _ = next_batch(2000)
                xa = sess.run([hidden5_anchor], feed_dict={x_a: xa})[0]
                xp = sess.run([hidden5_anchor], feed_dict={x_a: xp})[0]
                xn = sess.run([hidden5_anchor], feed_dict={x_a: xn})[0]
                satisfied = 0 
                for j in xrange(xa.shape[0]):
                    if np.linalg.norm(xa[j] - xp[j]) < np.linalg.norm(xa[j] - xn[j]):
                        satisfied += 1
                accuracy.append(float(satisfied) / 2000)
                print 'Iteration {0}: {1}'.format(iteration, accuracy[-1])
            if iteration % 500000 == 0 and iteration > 0:
                saver.save(sess, FLAGS.checkpoint_dir + '/embed_model_{0}.ckpt'.format(iteration))
            _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict())
            losses.append(loss_val)
            iteration += 1
            
        saver.save(sess, FLAGS.checkpoint_dir + '/embed_model_{0}.ckpt'.format(iteration))
        print 'Saved model checkpoint {0}.'.format(FLAGS.checkpoint_dir + '/embed_model_{0}.ckpt'.format(iteration))
        np.savetxt(FLAGS.summaries_dir + '/losses.txt', np.array(losses))
        print 'Saved losses file {0}'.format(FLAGS.summaries_dir + '/losses.txt')
        np.savetxt(FLAGS.summaries_dir + '/accuracy.txt', np.array(accuracy))
        print 'Saved accuracy file {0}'.format(FLAGS.summaries_dir + '/accuracy.txt')
       
def main(argv):
    train(argv)

if __name__ == '__main__':
    tf.app.run()
