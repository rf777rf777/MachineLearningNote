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

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)