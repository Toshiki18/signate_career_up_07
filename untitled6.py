じょ#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 17:56:28 2023

@author: oyamatoshiki
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import unicodedata as ud
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from sklearn.utils.class_weight import compute_sample_weight


#from imblearn.over_sampling import SMOTE


df = pd.read_csv('train.csv', encoding='utf-8')

df_fuel = df.groupby("fuel")["price"].mean()
train = pd.read_csv("train.csv", encoding='utf-8')
test = pd.read_csv("test.csv", encoding='utf-8')


train['manufacturer'] = train['manufacturer'].str.normalize('NFKC').str.upper().str.replace(' ', '')
test['manufacturer'] = test['manufacturer'].str.normalize('NFKC').str.upper().str.replace(' ', '')
train['manufacturer'] = train['manufacturer'].apply(lambda x: ud.normalize('NFKD', x).encode('ascii', 'ignore').decode())
test['manufacturer'] = test['manufacturer'].apply(lambda x: ud.normalize('NFKD', x).encode('ascii', 'ignore').decode())
# ここは手打ちです
replacements = {
    'NISAN': 'NISSAN',
    'VLKSWAGEN': 'VOLKSWAGEN',
    'SUBRU': 'SUBARU',
    'TOYOT': 'TOYOTA',
    'LEXU': 'LEXUS',
    'HRYSLER': 'CHRYSLER',
    'STURN': 'SATURN',
    'CURA': 'ACURA'
}

# Replace the values
train['manufacturer'] = train['manufacturer'].replace(replacements)
test['manufacturer'] = test['manufacturer'].replace(replacements)



def Setyear(df):
    if df['year'] > 2023:
        df['year'] = df['year'] - 1000
    return df

def Setmanufacturer(df):
    df['manufacturer'] = ud.normalize('NFKC', df['manufacturer'] )
    df['manufacturer'] = df['manufacturer'].strip()
  

    if df['manufacturer'].isupper():
        df['manufacturer'] = df['manufacturer'].lower()
    #if ret == 'F':
        #df['manufacturer'] = mojimoji.han_to_zen(df['manufacturer'])
    return df

def Setmeter(df):
    if df['odometer'] < 1000 and df['odometer'] >= -1:
        df['odometer'] = 1000
    elif df['odometer'] <= -2:
        df['odometer'] = - df['odometer']
    elif df['odometer'] > 400000:
        df['odometer'] = 400000
    return df





train = train.apply(Setyear, axis = 1)
train = train.apply(Setmanufacturer, axis = 1)
train = train.apply(Setmeter, axis = 1)
#train = train.apply(Setfuel, axis = 1)



test = test.apply(Setyear, axis = 1)
test = test.apply(Setyear, axis = 1)
test = test.apply(Setmeter, axis = 1)

'''
plt.hist(train['price'], label = ['row'])
plt.legend()
'''
class_1 = []


for i in range(len(train)):
    if train['price'][i] < 1050:
        class_1.append(1)
    elif train['price'][i] < 1100:
        class_1.append(2)
    elif train['price'][i] < 1300:
        class_1.append(3)
    elif train['price'][i] < 1400:
        class_1.append(4)
    elif train['price'][i] < 1500:
        class_1.append(5)
    elif train['price'][i] < 1600:
        class_1.append(6)
    elif train['price'][i] < 1700:
        class_1.append(7)
    elif train['price'][i] < 1800:
        class_1.append(8)
    elif train['price'][i] < 1900:
        class_1.append(9)
    elif train['price'][i] < 2000:
        class_1.append(10)
    #elif train['price'][i] < 3000:
     #   class_1.append(11)
    #elif train['price'][i] < 4000:
    #    class_1.append(12)
    #elif train['price'][i] < 5000:
    #    class_1.append(13)
    #elif train['price'][i] < 6000:
    #    class_1.append(14)
    #elif train['price'][i] < 7000:
    #    class_1.append(15)
    #elif train['price'][i] < 8000:
    #    class_1.append(16)
    #elif train['price'][i] < 9000:
    #    class_1.append(17)
    #elif train['price'][i] < 10000:
    #    class_1.append(18)  
    #elif train['price'][i] < 20000:
     #   class_1.append(19)
    #elif train['price'][i] < 30000:
    #    class_1.append(20)
    #elif train['price'][i] < 40000:
    #    class_1.append(21)
    #elif train['price'][i] < 50000:
    #    class_1.append(22)
    #elif train['price'][i] < 60000:
    #    class_1.append(23)
    #elif train['price'][i] < 65000:
    #    class_1.append(24)
    #elif train['price'][i] < 70000:
    #    class_1.append(25)
    #elif train['price'][i] < 75000:
    #    class_1.append(26) 
    #elif train['price'][i] < 80000:
    #    class_1.append(27)
    #elif train['price'][i] < 85000:
    #    class_1.append(28)
    #elif train['price'][i] < 90000:
    #    class_1.append(29)
    else:
        class_1.append(30)
        
train['class'] = class_1







train_2 = train[['year', 'manufacturer', 'condition', 'fuel', 'odometer', 'title_status', 'transmission' , 'size', 'type', 'price', 'class']]

train_3 = train_2.dropna(how='any')

x_res = train_3[['year', 'manufacturer', 'condition', 'fuel', 'odometer', 'title_status', 'transmission' , 'size', 'type', 'price', 'class']]
y_class = train_3['class']


#LightGBM
from sklearn.metrics import mean_squared_error # モデル評価用(平均二乗誤差)
from sklearn.metrics import r2_score # モデル評価用(決定係数)
import lightgbm as lgb
#df_car3.copy()
y = x_res["price"]
#LIGHTGBMはカテゴリ値も使えるので・・・objectタイプをカテゴリ値へ変換
x_res['year']=train['year'].astype('category')
x_res['manufacturer']=train['manufacturer'].astype('category')
x_res['condition']=train['condition'].astype('category')
x_res['fuel']=train['fuel'].astype('category')
x_res['odometer']=train['odometer'].astype('category')
x_res['title_status']=train['title_status'].astype('category')
x_res['transmission']=train['transmission'].astype('category')
x_res['size']=train['size'].astype('category')
x_res['class']=train['class'].astype('category')

test['year']=test['year'].astype('category')
test['manufacturer']=test['manufacturer'].astype('category')
test['condition']=test['condition'].astype('category')
test['fuel']=test['fuel'].astype('category')
test['odometer']=test['odometer'].astype('category')
test['title_status']=test['title_status'].astype('category')
test['transmission']=test['transmission'].astype('category')
test['size']=test['size'].astype('category')


x = x_res[['year', 'manufacturer', 'condition', 'fuel', 'odometer', 'title_status', 'transmission' , 'size', 'class']]
test_x = test[['year', 'manufacturer', 'condition', 'fuel', 'odometer', 'title_status', 'transmission' , 'size']]

x_column = ['year', 'manufacturer', 'condition', 'fuel', 'odometer', 'title_status', 'transmission' , 'size']

x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state = 82)

train_weight = compute_sample_weight(class_weight='balanced', y = x_train['class']).astype('float32')

x_train = x_train.drop('class', axis = 1)
x_valid = x_valid.drop('class', axis = 1)

print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)

lgb_train = lgb.Dataset(x_train, y_train, weight=train_weight)
lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)
#パラメーターの設定
params = {
    'bagging_fraction': 0.48588631957663897,
    'bagging_freq': 3,
    'boosting_type': 'gbdt',
    'categorical_column': x_column,
    'early_stopping_round': 50,
    'feature_fraction': 0.5,
    'feature_pre_filter': 'False',
    'lambda_l1': 7.044626094004133e-07,
    'lambda_l2': 0.00015833744596068677,
    'metric': 'l1',
    'min_child_samples': 100,
    'num_iterations': 200,
    'num_leaves': 6,
    'objective': 'mape',
    'verbosity': -1
}

model = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_eval,
                early_stopping_rounds=50
               )

#N = 0.52
pred = model.predict(x_valid)







score = mean_absolute_percentage_error(y_valid, pred)
print(score*100)


#横軸が予測結果、縦軸が正解価格です
plt.scatter(pred, y_valid)
plt.xlabel("predict")
plt.ylabel("price")
plt.show()

df_2 = pd.DataFrame({'pred' : pred})

pred_sub = pd.concat([x_valid, y_valid], axis = 1)
pred_sub = pred_sub.reset_index()
pred_sub = pd.concat([pred_sub, df_2], axis = 1)
#df_error = pd.DataFrame({'error' :[abs(pred_sub['price'] - pred_sub['pred'])]})
#pred_sub = pd.concat([pred_sub, df_error, abs(pred_sub['price'] - pred_sub['pred']) / pred_sub['price'] * 100], axis = 1)
pred_sub['error'] = abs(pred_sub['price'] - pred_sub['pred'])
pred_sub['mape'] = abs(pred_sub['price'] - pred_sub['pred']) / pred_sub['price'] * 100

#plt.hist([pred_sub['price'], pred_sub['pred']], label = ['price', 'pred'], stacked=False)
plt.hist(pred_sub['mape'], label = 'mape')
plt.legend()
#submit_sample.csvを読み込みます。
submit = pd.read_csv("submit_sample.csv", header=None)

predict = model.predict(test_x)

submit[1] = predict

#submit.to_csv("submission_8_5_3.csv", index=False, header=None)





