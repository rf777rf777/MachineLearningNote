import tensorflow as tf
import numpy as np

#建立tensorflow結構開始

#建立100個隨機數
x_data = np.random.rand(100).astype(np.float32)

y_data =  x_data*0.1 + 0.3

#開始建立tensflow結構
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

#計算預測值與實際值的誤差
loss = tf.reduce_mean(tf.square(y-y_data))

#優化方法:這裡選擇梯度下降Gradient Descent 
#學習效率(一般來說小於1): 這裡設定0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)

#結構初始化
init = tf.initialize_all_variables()

#建立tensorflow結構結束

sess = tf.Session()
sess.run(init)  #很重要(要先激活神經網路)

for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		print(step,sess.run(Weights),sess.run(biases))

