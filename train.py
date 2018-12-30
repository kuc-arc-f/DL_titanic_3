# coding: utf-8
# layers保存を、追加。
#
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd
import time

import matplotlib.pyplot as plt
from optimizer import SGD, Adam
from multi_layer_net_extend import MultiLayerNetExtend
from keras.utils import np_utils

#
def get_subData(src ):
    sub=src
    sub["Age"] = src["Age"].fillna( src["Age"].median())
    sub = sub.dropna()
    sub["Embark_flg"] = sub["Embarked"]
    sub["Embark_flg"] = sub["Embark_flg"].map(lambda x: str(x).replace('C', '0') )    
    sub["Embark_flg"] = sub["Embark_flg"].map(lambda x: str(x).replace('Q', '1') )
    sub["Embark_flg"] = sub["Embark_flg"].map(lambda x: str(x).replace('S', '2') )
    sub.groupby("Embark_flg").size()
    # convert, num
    sub = sub.assign( Embark_flg=pd.to_numeric( sub.Embark_flg ))
    sub["Sex_flg"] = sub["Sex"].map(lambda x: 0 if x =='female' else 1)    
    return sub

# 標準化対応、学習。
# 学習データ
train_data = pd.read_csv("train.csv" )
test_data = pd.read_csv("test.csv" )
print( train_data.shape )
#print( train_data.head() )
#
# 前処理 ,欠損データ 中央値で置き換える
train2  = train_data[["PassengerId","Survived","Sex","Age" , "Embarked" ,"SibSp" ,"Parch" ]]
test2   = test_data[ ["PassengerId"           ,"Sex","Age" , "Embarked" ,"SibSp" ,"Parch" ]]
#
age_mid=train2["Age"].median()
#print(age_mid )
print(train2.info() )
print(train2.head(10 ) )
#train2 = train2.dropna()
#train2["Embark_flg"] = train2["Embarked"].map(lambda x: str(x).replace('C', '0') )

train_sub =get_subData(train2 )
test_sub =get_subData(test2 )
print(train_sub.info() )
print(test_sub.info() )
#quit()

# 説明変数と目的変数
x_train= train_sub[["Sex_flg","Age" , "Embark_flg" ,"SibSp" ,"Parch" ]]
y_train= train_sub['Survived']
x_test = test_sub[["Sex_flg","Age" , "Embark_flg" ,"SibSp" ,"Parch" ]]

#conv
num_max_y=10
colmax_x =x_train[ "Age" ].max()
#x_train = x_train / colmax_x
#print(x_train[: 10 ])
#quit()

#print("#check-df")
#col_name="Age"
#print(x_train[ col_name ].max() )
#print(x_train[ col_name ].min() )
#quit()
#np
x_train = np.array(x_train, dtype = np.float32).reshape(len(x_train), 5)
y_train = np.array(y_train, dtype = np.float32).reshape(len(y_train), 1)
#正解ラベルをOne-Hot表現に変換
#y_train = y_train / num_max_y
#x_test  = x_test / num_max_y
y_train=np_utils.to_categorical(y_train, 2)
#
# 学習データとテストデータに分ける
print(x_train.shape, y_train.shape )
print(x_test.shape  )

# train
max_epochs = 50
#train_size = 3000
train_size = x_train.shape[ 0]
batch_size = 100
learning_rate = 0.01
iters_num = 10 * 1000   # 繰り返しの回数を適宜設定する    
#iter_per_epoch = max(train_size / batch_size, 1)
iter_per_epoch = 1000
print(iter_per_epoch )
#weight_init_std="relu"
weight_init_std="sigmoid"
#quit()


# train
global_start_time = time.time()
#
network = MultiLayerNetExtend(input_size=5
                                , hidden_size_list= [100, 100, 100, 100, 100] , output_size=2, 
                                weight_init_std=weight_init_std, use_batchnorm=True)
optimizer = SGD(lr=learning_rate)
#    hidden_size_list=[10, 10, 10, 10, 10 ]
bn_train_acc_list = []    
epoch_cnt = 0

for i in range(iters_num):        
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]
    #
    grads = network.gradient(x_batch, y_batch)
    optimizer.update( network.params, grads)

    if i % iter_per_epoch == 0:
        bn_train_acc = network.accuracy(x_train, y_train)
        bn_train_acc_list.append(bn_train_acc)
        print("epoch:" + str(epoch_cnt) + " | "  + " acc: " + str(bn_train_acc))
        epoch_cnt += 1
# pred
pred =network.predict(x_train)
#    print(pred[: 10])
#quit()
for item in pred[: 10]:
    y = np.argmax(item )
    #y = np.argmax(item, axis=1)
    #print(item)
#        print(y)
#    return train_acc_list, bn_train_acc_list
# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")
network.save_layers("p_layers.pkl")
print("Saved Network ,layers!")
    #
#bn_train_acc_list = proc_train("sigmoid")
print ('time : ', time.time() - global_start_time)
#quit()

#plt
a1=np.arange(len(bn_train_acc_list) )
plt.plot(a1 , bn_train_acc_list , label = "predict")
plt.legend()
plt.grid(True)
plt.title("titanic")
plt.xlabel("x")
plt.ylabel("acc")
plt.show()
quit()
