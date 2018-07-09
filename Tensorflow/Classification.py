'''
Classification 分類問題 (相對於線性回歸有連續分布的值)
機器學習中的監督學習(supervised learning)問題 大致可以分成 Regression (回歸)和 Classification(分類)這兩種
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#如果電腦裡沒有mnist 會去網路上下載
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

import tensorflow as tf 

#activation_function = None 沒有激勵函數 相當於線性
def add_layer(inputs,in_size,out_size,activation_function=None):
	#權重
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	#偏差 (機器學習中建議偏差不要為0 所以我們在這裡加上0.1)
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
	#matmul：矩陣乘法
	Wx_plus_b = tf.matmul(inputs,Weights) + biases

	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)

	return outputs

def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre = sess.run(prediction,feed_dict={xs:v_xs})
	#對比預測值與真實值最大數字1 的位置
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
	return result


#不規定有幾個Sample:None , 但每個Sample的大小784(28x28個相素點) ,因為手寫辨識範例圖片就是28x28
xs = tf.placeholder(tf.float32,shape=[None,784]) #28*28
#每張圖片都是一個數字,所以我們輸出的數字是0~9 共10類
ys = tf.placeholder(tf.float32,shape=[None,10])

#softmax激勵函數 一般是來處理Classification的 所以這裡用它
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

#相當於loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.initialize_all_variables())

for i in range(1000):
	batch_xs,batch_ys = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
	if i % 50 == 0:
		print(compute_accuracy(mnist.test.images,mnist.test.labels))