import tensorflow as tf

a=tf.constant([[1,2,3],[4,5,6]], name='a')
b=tf.constant([[1],[2]], name='b')
add_op=a+b

with tf.Session() as sess:
    print(sess.run(add_op))

print(a.shape)