import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

W=tf.Variable(np.random.randn(),)
b=tf.Variable(np.random.randn(),)
x=tf.placeholder(tf.float32)
linear_model=W*x+b;

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
#sess.run(linear_model,{x:[1,2,3,4]}))

y=tf.placeholder(tf.float32)
sq_deltas=tf.square(linear_model-y)
loss=tf.reduce_sum(sq_deltas)

optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
x_train=np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
7.042,10.791,5.313,7.997,5.654,9.27,3.1])
y_train=np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
2.827,3.465,1.65,2.904,2.42,2.94,1.3])

sess.run(init)

for i in range(1000):
    for (X, Y) in zip(x_train, y_train):
        sess.run(train, feed_dict={x: X, y: Y})

print(sess.run([W, b, loss], {x:x_train,y:y_train}))

plt.plot(x_train,y_train,'ro',label='Original data')
plt.plot(x_train,sess.run(W) * x_train + sess.run(b),label='Fitted Line')
plt.legend()
plt.show()

