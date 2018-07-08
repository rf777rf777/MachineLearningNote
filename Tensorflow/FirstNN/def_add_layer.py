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