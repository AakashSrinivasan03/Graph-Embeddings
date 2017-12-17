import tensorflow as tf
import numpy as np

g = tf.Variable(tf.truncated_normal([10, 10]))

h = tf.placeholder(shape=[10, 10], dtype=tf.float32, name='Y')

f = tf.trace(g)

Sess = tf.Session()

initialiser = tf.global_variables_initializer()

Sess.run(initialiser)

optimiser = tf.train.AdamOptimizer(learning_rate=0.02).minimize(f)

for i in range(100):

    rand_arr = np.random.random((10, 10))

    loss, _ = Sess.run([f, optimiser], feed_dict={h: rand_arr})

    print(' f ' + str(loss))




