import tensorflow as tf
import matplotlib.pyplot as plt

W=tf.Variable([0.3],dtype=tf.float32)
b=tf.Variable([-0.3],dtype=tf.float32)
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
x_train=[1,2,3,4]
y_train=[0,-1,-2,-3]
sess.run(init)
for i in range(500):
    sess.run(train,{x:x_train,y:y_train})

#curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train,y:y_train})
#print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

plt.plot(x_train,y_train,'ro',label='Original data')
plt.plot(x_train,sess.run(W) * x_train + sess.run(b),label='Fitted Line')
plt.legend()
plt.show()
#print(sess.run([W,b,loss]),{x:[1,2,3,4],y:[0,-1,-2,-3]})
