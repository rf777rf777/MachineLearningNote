# -*- coding: UTF-8 -*-

import tensorflow as tf

#Variable(初始值,名稱)
state = tf.Variable(0,name = 'counter')
#print(state.name)

one = tf.constant(1)
new_value = tf.add(state , one)
update = tf.assign(state,new_value)

init = tf.initialize_all_variables() #如果有定義變量 一定要用這行

with tf.Session() as sess:
	sess.run(init)
	for i in range(3):
		sess.run(update)
		print(sess.run(state))
		