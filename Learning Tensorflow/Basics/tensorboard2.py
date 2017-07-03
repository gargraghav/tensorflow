import tensorflow as tf

a = tf.add(1, 2, name="a")
b = tf.multiply(a, 3, name="b")
c = tf.add(4, 5, name="c")
d = tf.multiply(c, 6, name="d")
e = tf.multiply(4, 5, name="e")
f = tf.div(c, 6, name="f")
g = tf.add(b, d,name="g")
h = tf.multiply(g, f, name="h")

with tf.Session() as sess:
    writer=tf.summary.FileWriter("/home/raghav/PycharmProjects/tensorflow/tensorboard/output2",sess.graph)
    print(sess.run(h))
    writer.close()