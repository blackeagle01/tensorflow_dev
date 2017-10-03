import tensorflow as tf
import numpy as np

class NeuralNet(object):
	def __init__(self,input_dim):
		self.input_dim=input_dim
		self.currentstate=None
		self.currentsize=input_dim
		self.synapses=[]
		self.biases=[]
		self.activation=[]



	def addDenseLayer(self,size,activation):
		w=tf.Variable(np.random.randn(self.currentsize,size ),dtype=tf.float32)
		b=tf.Variable(np.random.random(size),dtype=tf.float32)
		self.currentsize=size
		self.synapses.append(w)
		self.biases.append(b)
		self.activation.append(activation)


	def predict(self,inp):
		currentstate=tf.Variable(inp,dtype=tf.float32)
		for w,b,a in zip(self.synapses,self.biases,self.activation):
			currentstate=tf.matmul(currentstate,w)+b
			if a=='relu':
				currentstate=tf.nn.relu(currentstate)
			else:
				currentstate=tf.nn.softmax(currentstate)

		return tf.Variable(currentstate,dtype=tf.float32)



sess=tf.Session()
init=tf.global_variables_initializer()
sess=tf.Session()
init=tf.global_variables_initializer()
nn=NeuralNet(3)
nn.addDenseLayer(5,activation='relu')
nn.addDenseLayer(5,activation='relu')
nn.addDenseLayer(3,activation='softmax')



print(sess.run(nn.predict([[1,3,2]])))
