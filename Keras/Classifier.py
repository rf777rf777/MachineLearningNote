import numpy as np 
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential #用來一層一層的建立神經層
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28):60000筆資料 每筆28x28個像素, y shape (10,000, ):10000個標籤
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#data pre-processing
X_train = X_train.reshape(X_train.shape[0],-1)/255 #normalize 標準化至0~1範圍間
X_test = X_test.reshape(X_test.shape[0],-1)/255 #normalize 標準化至0~1範圍間
#為什麼要除以255? 因為每個像素(pixel)的值都介於0~255 除以255將其值標準化至0~1之間

#np_utils.to_categorical: one-hot encoding將數值轉變成向量
#這裡為0~9的值轉變成 大小為10的向量
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)


# Another way to build your neural net
# 兩層Dense 第一層輸出32個特徵值 第二層輸出10個特徵值
model = Sequential([
			Dense(32,input_dim = 784),
			Activation('relu'),
			Dense(10),
			Activation('softmax')
		])

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

# We add metrics to get more results you want to see
model.compile(
			optimizer = rmsprop,
			loss = 'categorical_crossentropy',
			metrics=['accuracy']
		)


print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=2, batch_size=32)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
