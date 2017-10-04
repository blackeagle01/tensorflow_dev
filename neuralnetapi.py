import tensorflow as tf
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from keras.utils import to_categorical


class NeuralNet(object):


	def __init__(self,input_dim):
		self.input_dim=input_dim
		self.currentstate=None
		self.currentsize=input_dim
		self.synapses=[]
		self.biases=[]
		self.activation=[]



	def addDenseLayer(self,size,activation):
		w=tf.Variable(np.random.randn(self.currentsize,size ),dtype=tf.float32,name='synapses')
		b=tf.Variable(np.random.random(size),dtype=tf.float32,name='biases')
		self.currentsize=size
		self.synapses.append(w)
		self.biases.append(b)
		self.activation.append(activation)





	def predict(self,inp):
		sess=tf.Session()
		currentstate=tf.Variable(inp,dtype=tf.float32,name='currentstate')
		for w,b,a in zip(self.synapses,self.biases,self.activation):
			currentstate=tf.matmul(currentstate,w)+b
			if a=='relu':
				currentstate=tf.nn.relu(currentstate)
			else:
				currentstate=tf.nn.softmax(currentstate)
		sess.run(tf.global_variables_initializer())
		return sess.run(currentstate)




	def fit(self,inputdata,outputs):
		sess=tf.Session()
		currentstate=tf.Variable(inputdata,dtype=tf.float32,name='currentstate')
		for i in range(len(self.synapses)):
			currentstate=tf.matmul(currentstate,self.synapses[i])+self.biases[i]
			if self.activation[i]=='relu':
				currentstate=tf.nn.relu(currentstate)
			else:
				pass

		loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=currentstate,labels=outputs))
		train=tf.train.AdamOptimizer(0.005).minimize(loss)
		sess=tf.Session()
		sess.run(tf.global_variables_initializer())
		for i in range(1000):
			#print("Epoch ", i+1)
			sess.run(train)
			#print("Loss : ",sess.run(loss))
			#time.sleep(0.01)




if __name__=="__main__":

	d=load_iris()
	X_train,X_test,Y_train,Y_test=train_test_split(d.data,d.target)
	Y_train=to_categorical(Y_train)
	Y_test=to_categorical(Y_test)

	sess=tf.Session()
	init=tf.global_variables_initializer()
	sess.run(init)
	nn=NeuralNet(4)
	nn.addDenseLayer(6,activation='relu')
	nn.addDenseLayer(6,activation='relu')
	nn.addDenseLayer(3,activation='softmax')
	nn.fit(X_train,Y_train)
	pred=nn.predict(X_train[:4])
	sess.run(init)
	print(np.round(pred),Y_train[:4])

