import tensorflow as tf
import tensorflow as tf

#Point 1
x1 = tf.constant(2, dtype=tf.float32)
y1 = tf.constant(9, dtype=tf.float32)
p1 = tf.stack([x1, y1])

#Point 2
x2 = tf.constant(-1, dtype=tf.float32)
y2 = tf.constant(3, dtype=tf.float32)
p2 = tf.stack([x2, y2])

X = tf.transpose(tf.stack([p1, p2]))

B= tf.ones((1,2), dtype=tf.float32)

para = tf.matmul(B,tf.matrix_inverse(X))

with tf.Session() as sess:
    A=sess.run(para)

b = 1 / A[0][1]
a = -b * A[0][0]

print("Equation: y = {a}x + {b}".format(a=a, b=b))