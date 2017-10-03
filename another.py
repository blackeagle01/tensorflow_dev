from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import numpy as np
data=make_regression(100,1)
a,b,c,d=train_test_split(data[0],data[1])


w=np.random.random((1,1))
w=tf.Variable(w,dtype='float32')
x=tf.placeholder(tf.float32)
b=tf.Variable([0.9],dtype='float32')
model=tf.matmul(x,w)+b

sess=tf.Session()
loss=tf.reduce_mean(tf.square(model-c))
train=tf.train.AdamOptimizer(0.003).minimize(loss)
sess.run(tf.global_variables_initializer())
for _ in range(1000):
	sess.run(train,{x:a})
print(sess.run(loss,{x:a}))