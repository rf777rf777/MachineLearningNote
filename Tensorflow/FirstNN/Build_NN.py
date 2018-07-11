import def_add_layer as layerAdder
import tensorflow as tf 
import numpy as np

#讓結果可視化
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x_data = np.linspace(-1,1,300)[:,np.newaxis]

#噪點 使之更像真實數據
noise = np.random.normal(0,0.05,x_data.shape)

y_data = np.square(x_data) - 0.5 + noise

#x_data屬性只有1(假設只有1個神經元)
xs = tf.placeholder(tf.float32,[None,1])

#y_data屬性只有1(假設只有1個神經元)
ys = tf.placeholder(tf.float32,[None,1])
'''
	建立一個3層神經NN:
	輸入層有1個神經元 
	隱藏層有10個神經元 
	輸出層有1個神經元
'''

#隱藏層 輸入1(輸入層1個神經元) 輸出10(隱藏層要10個神經元)
#激勵函數嘗試使用relu	
#layer1 = def_add_layer.add_layer(x_data,1,10,activation_function = tf.nn.relu)
layer1 = layerAdder.add_layer(xs,1,10,activation_function = tf.nn.relu)

#輸出層 輸入10(隱藏層要10個神經元) 輸出1(輸出層1個神經元)
#假設沒有激勵函數	
#prediction = def_add_layer.add_layer(layer1,10,1,activation_function = None)
prediction = layerAdder.add_layer(layer1,10,1,activation_function = None)

#求平均誤差(預測值與真實值的差別)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
	reduction_indices=[1]))

#學習優化器
#學習效率learning_rate必須小於1 這裡假設0.1
#每一個train都縮小誤差minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)

	#建立一個圖片框
	fig = plt.figure() 
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x_data,y_data)

	#程式plot後不暫停
	plt.ion()
	plt.show()

	for i in range(1000):
		#loss跟xs與ys皆有關 feed_dict要輸入x_data,y_data
		sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
		if i % 50 == 0:
			#每50次顯示誤差 觀察誤差是否越來越小
			#print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
			
			try:
				#先刪掉第一條線
				ax.lines.remove(lines[0])
			except Exception:
				pass
			#prediction只跟xs有關 feed_dict要輸入x_data 用紅色的線'r-' 寬度為5
			prediction_value = sess.run(prediction,feed_dict={xs:x_data})
			#x軸的數據x_data y軸的數據prediction_value
			lines = ax.plot(x_data,prediction_value,'r-',lw=5)
			plt.pause(0.5)