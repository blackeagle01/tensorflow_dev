import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

#Create dummy regression problem and split it into training and testing data

data=make_regression(100,3)
X_train,X_test,Y_train,Y_test=train_test_split(data[0],data[1])

clf=LinearRegression()
clf.fit(X_train,Y_train)
print(clf.coef_) 

print('breakpoint\n')
#Create a machine learning Regression model

w=tf.Variable([0.5,0.9,0.7],dtype='float32')
b=tf.Variable([0.7],dtype='float32')
x=tf.placeholder(tf.float32)
y=w*x+b
loss=tf.reduce_sum(tf.square(y-Y_train))
'''hro'''
'''
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
print(sess.run(loss,{x:X_train}))
'''

optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(1000):
	sess.run(train,{x:X_train[i]})

print(sess.run([w,b],{x:X_train[i]}))