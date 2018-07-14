import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt 

import numpy as np 
np.random.seed(1337) # for reproducibility
from keras.models import Sequential #Sequential: 按順序建立的model
from keras.layers import Dense #Dense: 全連接層


#create some data
X = np.linspace(-1,1,200) #-1~1之間建立200個隨機數
np.random.shuffle(X) # randomize the data 隨機改變位置
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

#plot data
plt.scatter(X,Y)
#程式plot後不暫停
plt.ion()
plt.show()
plt.pause(1)

X_train,Y_train = X[:160],Y[:160] #first 160 data points
X_test,Y_test = X[160:],Y[160:] #last 40 data points

# build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(output_dim=1,input_dim=1)) #一個X一個Y(皆為1維)

# choose loss function and optimizing method
# mse:2次方誤差
model.compile(loss='mse',optimizer='sgd')


#training
print('Training------------')
for step in range(301):
	cost = model.train_on_batch(X_train,Y_train) #會回傳誤差值
	if step % 100 == 0:
		print('train cost:',cost) 


print('\nTesting------------')
cost = model.evaluate(X_test,Y_test,batch_size=40)
print('test cost:',cost) 

W,b = model.layers[0].get_weights()

print('Weights',W,'biases',b)


# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
plt.pause(10)
