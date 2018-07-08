# -*- coding: UTF-8 -*-

import tensorflow as tf

input1 = tf.placeholder(tf.float32, shape=None)
input2 = tf.placeholder(tf.float32, shape=None)

#output = input1*input2
output = tf.multiply(input1,input2)

inputA = tf.placeholder(tf.float32, shape=[2,1])
inputB = tf.placeholder(tf.float32, shape=[1,2])
outputAB = tf.matmul(inputA,inputB)

with tf.Session() as sess:
	#只執行一組運算
	output_Value = sess.run(output,feed_dict={input1:7,input2:2})
	print(output_Value)
	
	#執行多組運算
	output_Value , outputAB_Value = sess.run(
		[output,outputAB], #run them together
		feed_dict = {
			input1:1, input2:2,
			inputA:[[2],[2]], inputB:[[3,3]]
		})

	print(output_Value)
	print(outputAB_Value)