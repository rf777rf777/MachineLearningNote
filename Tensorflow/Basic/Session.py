# -*- coding: UTF-8 -*-

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

#矩陣乘法
product = tf.matmul(matrix1,matrix2)

#方法一
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

#方法二
with tf.Session() as sess:
	result2 = sess.run(product)
	print(result2)
