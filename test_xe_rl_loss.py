import os
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''

a = tf.constant(np.arange(6).reshape(2, 3) * 0.1)
b = tf.Variable(initial_value=np.ones((3, 1)))
c = tf.matmul(a, b)
l = tf.reduce_mean(c)

opt_1 = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(l)
opt_2 = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(l)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print 'step_1, b\n', sess.run(b)
print 'step_2, b\n', sess.run(b)

sess.run(opt_1)
print 'step_3, b\n', sess.run(b)
sess.run(opt_1)
print 'step_4, b\n', sess.run(b)

sess.run(opt_2)
print 'step_5, b\n', sess.run(b)
sess.run(opt_2)
print 'step_6, b\n', sess.run(b)