import tensorflow as tf

x=tf.placeholder("float", None)
y=x*2

with tf.Session() as sess:
    result=sess.run(y, feed_dict={x:[1,2,3]})
    print(result)

