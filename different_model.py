from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np

data = pd.read_csv('./家居厨房用品.csv')
# 2.切分数据输入：特征 输出：预测目标变量
def normalize(feature):
    data[feature] = (data[feature] - min(data[feature]))/(max(data[feature]) - min(data[feature]))
# ['潜力市场', '类目', '平均月销量', '月平均售价', '月搜索量', '过去30天搜索趋势', '过去90天搜索趋势', '竞争程度', '市场机会分数', '季节性', '旺季月份']
data.drop(['类目','季节性','旺季月份'],axis = 1,inplace=True)
data['月平均售价'] = data['月平均售价'].apply(lambda x:float(x.strip('$')))
data['过去30天搜索趋势'] = data['过去30天搜索趋势'].apply(lambda x:float(x.strip('%')))
data['过去90天搜索趋势'] = data['过去90天搜索趋势'].apply(lambda x:float(x.strip('%')))
data['转化率'] = data.apply(lambda x: x['平均月销量'] / x['月搜索量'], axis=1)
data['销售额'] = data.apply(lambda x: x['平均月销量'] * x['月平均售价'], axis=1)




model = {'中': 0, '低': 0, '非常低': 0, '非常高': 1, '高': 0}
data['竞争程度'] = data['竞争程度'].map(model)

for i in ['平均月销量', '月平均售价', '月搜索量', '过去30天搜索趋势', '过去90天搜索趋势','转化率','销售额']:
    normalize(i)

freq_dict = {}
for x in list(data['潜力市场'].values):
    freq_dict[x] = freq_dict.get(x, 0) + 1

data['潜力市场'] = data['潜力市场'].map(freq_dict).apply(lambda x:x/data['潜力市场'].__len__())


y = data.loc[:,'竞争程度']
X = data.loc[:,['潜力市场','平均月销量', '月平均售价', '月搜索量', '过去30天搜索趋势', '过去90天搜索趋势','转化率','市场机会分数','销售额']]





train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)


train_X_class = train_X.drop('市场机会分数',axis=1)
test_X_class = test_X.drop('市场机会分数',axis=1)

params = {
    'logging_level':'Verbose',
    'loss_function': 'MultiClass',
    # 'eval_metric': 'AUC',
}
clf_multiclass = CatBoostClassifier(**params)
train_data = train_X_class
train_label = train_y
test_data = test_X_class



clf_multiclass.fit(train_data,train_label,eval_set=(test_data,test_y),plot=True)
pred_y = clf_multiclass.predict(test_data)
test_y = list(test_y)
pred_y = list(pred_y.squeeze())

# model = {'非常高':0,'高':1,'中':2,'低':3,'非常低':4}
# test_y = list(pd.DataFrame(test_y)[0].map(model))
# pred_y = list(pd.DataFrame(pred_y)[0].map(model))


from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']    # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False           # 解决保存图像是负号'-'显示为方块的问题
import matplotlib.pyplot as plt
fea_ = clf_multiclass.feature_importances_
fea_name = clf_multiclass.feature_names_
dic = dict([(i,j) for i,j in zip(fea_,fea_name)])
f = zip(dic.keys(),dic.values())
dic = sorted(f)
fea_,fea_name = zip(*dic)
model = {'转化率':"Transfer rate", '月搜索量':"Search", '销售额':"GMV", '月平均售价':"Average price", '过去30天搜索趋势':"30 days search", '过去90天搜索趋势':"90 days search", '潜力市场':"Market", '平均月销量':"Sales"}
fea_name = pd.DataFrame(fea_name)[0].map(model)
plt.figure()
plt.barh(range(len(fea_)),fea_)
plt.yticks(range(len(fea_)),fea_name)
cat = []
print('Accuracy: ',sum([1 if i == j else 0 for i,j in zip(test_y,pred_y)])/len(pred_y))
auc_score = roc_auc_score(test_y,pred_y)
cat.append(sum([1 if i == j else 0 for i,j in zip(test_y,pred_y)])/len(pred_y))
cat.append(auc_score)






lg = LogisticRegression(C=1000)  # 默认使用L2正则化避免过拟合，C=1.0表示正则力度(超参数，可以调参调优)
lg.fit(train_data, train_label)

# 回归系数
print(lg.coef_)  # [[1.12796779  0.28741414  0.66944043 ...]]

# 进行预测
y_predict = lg.predict(test_data)
print('Accuracy: ',sum([1 if i == j else 0 for i,j in zip(test_y,y_predict)])/len(y_predict))

auc_score = roc_auc_score(test_y,y_predict)
lr = []
lr.append(sum([1 if i == j else 0 for i,j in zip(test_y,y_predict)])/len(y_predict))
lr.append(auc_score)




from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import load_model
#导入需要使用的库
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras.backend as K
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定只是用第三块GPU


def model_builder(x_data, y_data,x_test,y_test, class_num):
    if class_num == 2:  # 逻辑回归二分类
        layer0 = tf.keras.layers.Dense(64, input_shape=(x_data.shape[1],), activation='relu')
        layer1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(x_data.shape[1],))
        layer2 = tf.keras.layers.Dropout(rate=0.1)
        layer3 = tf.keras.layers.Dense(class_num, activation='softmax')
        model = tf.keras.Sequential([layer0,layer1,layer2,layer3])
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])  # 这里用二元的交叉熵作为二分类的损失函数
        model.evaluate(x_test, y_test)
    else:  # 多分类
        layer0 = tf.keras.layers.Dense(64, input_shape=(x_data.shape[1],), activation='relu')
        layer1 = tf.keras.layers.Dense(128, activation='relu', input_shape=(x_data.shape[1],))
        layer2 = tf.keras.layers.Dropout(rate=0.1)
        layer3 = tf.keras.layers.Dense(class_num, activation='softmax')
        model = tf.keras.Sequential([layer0,layer1,layer2,layer3])
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        model.evaluate(x_test, y_test)
    return model



data = pd.read_csv('./家居厨房用品.csv')
# 2.切分数据输入：特征 输出：预测目标变量
def encode(feature):
    model = dict(zip(sorted(list(set(data[feature]))), range(0, len(set(data[feature])))))
    data[feature] = data[feature].map(model)
def normalize(feature):
    data[feature] = (data[feature] - min(data[feature]))/(max(data[feature]) - min(data[feature]))
# ['潜力市场', '类目', '平均月销量', '月平均售价', '月搜索量', '过去30天搜索趋势', '过去90天搜索趋势', '竞争程度', '市场机会分数', '季节性', '旺季月份']
data.drop(['类目','季节性','旺季月份'],axis = 1,inplace=True)
data['月平均售价'] = data['月平均售价'].apply(lambda x:float(x.strip('$')))
data['过去30天搜索趋势'] = data['过去30天搜索趋势'].apply(lambda x:float(x.strip('%')))
data['过去90天搜索趋势'] = data['过去90天搜索趋势'].apply(lambda x:float(x.strip('%')))
data['转化率'] = data.apply(lambda x: x['平均月销量'] / x['月搜索量'], axis=1)
data['销售额'] = data.apply(lambda x: x['平均月销量'] * x['月平均售价'], axis=1)
model = {'中': 0, '低': 0, '非常低': 0, '非常高': 1, '高': 0}
data['竞争程度'] = data['竞争程度'].map(model)
encode('潜力市场')
for i in ['潜力市场','平均月销量', '月平均售价', '月搜索量', '过去30天搜索趋势', '过去90天搜索趋势','转化率','销售额']:
    normalize(i)
y = data.loc[:,'竞争程度']
X = data.loc[:,['潜力市场','平均月销量', '月平均售价', '月搜索量', '过去30天搜索趋势', '过去90天搜索趋势','转化率','市场机会分数','销售额']]
X, X_test, Y, Y_test = train_test_split(X, y, test_size=0.2)




class_num=2
gpu=False
output1="./data/output.h5"
x_data = np.array(X.drop('市场机会分数',axis=1))
y_data = np.array(Y).reshape(-1,1)
x_test = np.array(X_test.drop('市场机会分数',axis=1))
y_test = np.array(Y_test).reshape(-1,1)
try:
    y_data = to_categorical(y_data)  # 一维的分类转成多列
    y_data = pd.DataFrame(y_data)
    y_test = to_categorical(y_test)  # 一维的分类转成多列
    y_test = pd.DataFrame(y_test)
    if gpu:
        dataset, BATCH_SIZE, strategy = multiple_gpu_strategy(x_data, y_data)
        with strategy.scope():
            model = model_builder(x_data, y_data,x_test,y_test, class_num)
        model.fit(dataset.batch(BATCH_SIZE), verbose=False)
    else:
        pass
        # model = model_builder(x_data, y_data,x_test,y_test, class_num)
        # model.fit(x_data, y_data, epochs=10000)
except Exception:
    raise Exception("模型训练错误")
print("模型训练完成")
# if output1:
#     model.save(output1)
model = load_model(output1)

loss, accuracy = model.evaluate(x_test, y_test)
print('test loss: ', loss)
print('accuracy: ', accuracy)
y_predict = model.predict(x_test,batch_size = 1)
def decode(y):
    y_ = []
    for j in range(len(y)):
        y_.append([i for i in range(len(y[j])) if y[j][i] == max(y[j])])
    return list(np.array(y_).squeeze())

y_test = decode(np.array(y_test))
y_predict = decode(y_predict)



print('Accuracy: ',sum([1 if i == j else 0 for i,j in zip(y_test,y_predict)])/len(y_predict))

auc_score = roc_auc_score(y_test,y_predict)
nn = []
nn.append(sum([1 if i == j else 0 for i,j in zip(y_test,y_predict)])/len(y_predict))
nn.append(auc_score)


import matplotlib.pyplot as plt

size = 2
x = np.arange(size)

total_width, n = 0.4, 3
width = total_width / n
x = x - (total_width - width) / 3
plt.figure()
plt.bar(x, lr,  width=width, label='Logistic Regression')
plt.bar(x + width, cat, width=width, label='Decision Tree')
plt.bar(x + 2 * width, nn, width=width, label='Neural Network')

plt.xticks([0,1], ['Accuracy', 'AUC'])
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.xlim(-0.2,1.3)
plt.ylim(0,1)
plt.legend()
plt.show()