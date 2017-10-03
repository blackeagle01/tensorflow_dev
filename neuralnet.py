import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical
import time
d=load_iris()
targets=to_categorical(d.target)
X_train,X_test,Y_train,Y_test=train_test_split(d.data,targets)

inputlayer=tf.placeholder(dtype=tf.float32)
syn0=tf.Variable(np.random.randn(4,5),dtype=tf.float32)
b0=tf.Variable(np.random.random(5),dtype=tf.float32)
hlayer=tf.matmul(inputlayer,syn0)+b0
hlayer_=tf.nn.relu(hlayer)
syn1=tf.Variable(np.random.randn(5,5),dtype=tf.float32)
b1=tf.Variable(np.random.random(5),dtype=tf.float32)
hlayer1=tf.matmul(hlayer_,syn1)+b1
hlayer1_=tf.nn.relu(hlayer1)
syn2=tf.Variable(np.random.randn(5,3),dtype=tf.float32)
b2=tf.Variable(np.random.random(3),dtype=tf.float32)
final=tf.matmul(hlayer1_,syn2)+b2
output=tf.nn.softmax(final)
actualoutput=tf.placeholder(tf.float32)
#loss= -tf.reduce_mean(actualoutput*tf.log(output+0.00001)+(1- actualoutput)*tf.log(1- output+0.0001) )
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final,labels=actualoutput))
train=tf.train.AdamOptimizer(0.01).minimize(loss)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2000):
	print('Epoch :',i+1)
	print(sess.run(loss,{inputlayer:X_train,actualoutput:Y_train}))
	sess.run(train,{inputlayer:X_train,actualoutput:Y_train})
	time.sleep(.01)

print("Calculating loss on Test Data")
print(sess.run(loss,{inputlayer:X_test,actualoutput:Y_test}))
print(np.round(sess.run(output,{inputlayer:X_test[:10]})))
print(Y_test[:10])