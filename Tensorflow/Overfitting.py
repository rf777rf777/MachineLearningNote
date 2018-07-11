import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits = load_digits()
#特徵矩陣 (觀測值)
X = digits.data
#標籤矩陣 (目標值)
y = digits.target
#標籤矩陣二值化
y = LabelBinarizer().fit_transform(y)

#為避免過擬合 採用交叉驗證 驗證集佔訓練集30%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)


#activation_function = None 沒有激勵函數 相當於線性
def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name = 'layer{}'.format(n_layer)

    #權重
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    #偏差 (機器學習中建議偏差不要為0 所以我們在這裡加上0.1)
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    #matmul：矩陣乘法
    Wx_plus_b = tf.matmul(inputs,Weights) + biases

    #Dropout主功能
    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    #多加的 其實不需要 為了防止只有tf.summary.scalar 執行出現的錯誤
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

#for dropout : 一直保持多少的結果不被Dropout
keep_prob = tf.placeholder(tf.float32)

xs = tf.placeholder(tf.float32,[None,64]) #8x8
ys = tf.placeholder(tf.float32,[None,10])

#新增隱藏層 (輸出100個神經元只是為了看出overfitting的問題)
#改成50是防止計算值太大變NAN : 
#在輸出結果需要sum全部weights*xs,越多neurones(神經元)越多weights, sum就會越大, sum太大會爆炸計算出來是NAN. 
#有鑑於此 一種方法是直接減少 neurones, 
#一種是初始 weights 的時候 將standard distribution縮小.﻿

layer1 = add_layer(xs,64,50,'layer1',activation_function=tf.nn.tanh)

#新增輸出層
prediction = add_layer(layer1,50,10,'layer2',activation_function=tf.nn.softmax)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

tf.summary.scalar('loss',cross_entropy)
#學習效率=0.6
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
test_writer = tf.summary.FileWriter("logs/test",sess.graph)


sess.run(tf.initialize_all_variables())

for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})
    if i % 50 == 0:
        #keep_prob=1 (100%)：紀錄的時候不要drop任何東西
        train_result = sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1}) 
        test_result = sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result,i)

#執行Tensorboard : tensorboard --logdir logs