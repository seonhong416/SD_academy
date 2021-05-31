import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./4.15/cardio_train.csv', sep = ';')
print(df)
print(df.info())
print(df.describe())
print(df.isna().sum())
## 결측치가 없다.

df_1 = df.copy()

df_1 = df_1.drop('id', axis = 1)
## id칼럼 삭제

def age(x) :
    x = x/365 
    return int(x)

df_1['age_year'] = df['age'].apply(age)
## 'age' 칼럼이 day로 되어 있어서 나이로 바꿈.


df_1['age_year'].value_counts()



df_1 = df_1.drop('age', axis = 1)
## day로 되어있는 age칼럼 삭제

df_1.loc[df_1['age_year']<40, 'age_bin'] = 0 
df_1.loc[(df_1['age_year'] >= 40) & (df_1['age_year'] < 50), 'age_bin'] = 1 
df_1.loc[(df_1['age_year'] >= 50) & (df_1['age_year'] < 60), 'age_bin'] = 2
df_1.loc[df_1['age_year'] >= 60, 'age_bin'] = 3
## 나이별로 범주화 했음

df_1 = df_1.drop('age_year', axis = 1)
## age_year칼럼 삭제

## BMI 지수 : 몸무게(kg) / 키(m) 의 제곱
df_1['BMI'] = df_1['weight'] / (df_1['height']/100)**2

fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(data = df_1['BMI'])
plt.show()
## bmi지수 사분위수 그래프 그려봄


## IQR(BMI)


BMI_q1 = np.percentile(df_1['BMI'], 25)
BMI_q3 = np.percentile(df_1['BMI'], 75)

BMI_IQR = BMI_q3 - BMI_q1
print(BMI_IQR)

df_1_IQR = df_1[(df_1['BMI'] >= (BMI_q1 - (1.5 * BMI_IQR))) & (df_1['BMI'] <= (BMI_q3 + (1.5 * BMI_IQR)))]
print(df_1_IQR)

fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(data = df_1_IQR['BMI'])
plt.show()

df_1_IQR.loc[df_1_IQR['BMI'] < 18.5, 'BMI_bin'] = 0
df_1_IQR.loc[(df_1_IQR['BMI'] >= 18.5) & (df_1_IQR['BMI'] < 25), 'BMI_bin'] = 1
df_1_IQR.loc[(df_1_IQR['BMI'] >= 25) & (df_1_IQR['BMI'] < 30), 'BMI_bin'] = 2
df_1_IQR.loc[df_1_IQR['BMI'] >= 30, 'BMI_bin'] = 3
## BMI지수 범주화 했음

df_1_IQR = df_1_IQR.drop('BMI', axis = 1)


## ap_hi IQR 
## 혈압에 관한 부분도 이상치가 너무 심해서 다시 IQR 적용했습니다.
fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(data = df_1_IQR['ap_hi'])
plt.show()


ap_q1 = np.percentile(df_1_IQR['ap_hi'], 25)
ap_q3 = np.percentile(df_1_IQR['ap_hi'], 75)

ap_IQR = ap_q3 - ap_q1
print(ap_IQR)

df_1_IQR2 = df_1_IQR[(df_1_IQR['ap_hi'] >= (ap_q1 - (1.5 * ap_IQR))) & (df_1_IQR['ap_hi'] <= (ap_q3 + (1.5 * ap_IQR)))]
print(df_1_IQR2)

fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(data = df_1_IQR2['ap_hi'])
plt.show()


## ap_lo IQR

fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(data = df_1_IQR2['ap_lo'])
plt.show()

ap_q1 = np.percentile(df_1_IQR2['ap_lo'], 25)
ap_q3 = np.percentile(df_1_IQR2['ap_lo'], 75)

ap_IQR = ap_q3 - ap_q1
print(ap_IQR)

df_1_IQR3 = df_1_IQR2[(df_1_IQR2['ap_lo'] >= (ap_q1 - (1.5 * ap_IQR))) & (df_1_IQR2['ap_lo'] <= (ap_q3 + (1.5 * ap_IQR)))]
print(df_1_IQR3)

fig, ax = plt.subplots(figsize = (10, 5))
sns.boxplot(data = df_1_IQR3['ap_lo'])
plt.show()


fig, ax = plt.subplots(figsize = (15, 7), nrows = 2, ncols = 2)
sns.boxplot(data = df['ap_hi'], ax = ax[0][0])
sns.boxplot(data = df_1_IQR3['ap_hi'], ax = ax[0][1])
sns.boxplot(data = df['ap_lo'], ax = ax[1][0])
sns.boxplot(data = df_1_IQR3['ap_lo'], ax = ax[1][1])

ax[0][0].set(title = "ap_hi")
ax[0][1].set(title = "ap_hi_trans")
ax[1][0].set(title = "ap_lo")
ax[1][1].set(title = "ap_lo_trans")

plt.show()

df_1_IQR3.loc[(df_1_IQR3['ap_hi'] < 120) & (df_1_IQR3['ap_lo'] < 80), 'ap_bin'] = 0
df_1_IQR3.loc[((df_1_IQR3['ap_hi']>= 120) & (df_1_IQR3['ap_hi']< 140)) | ((df_1_IQR3['ap_lo']>= 80) & (df_1_IQR3['ap_lo'] < 90)), 'ap_bin'] = 1
df_1_IQR3.loc[((df_1_IQR3['ap_hi']>= 140) & (df_1_IQR3['ap_hi']< 160)) | ((df_1_IQR3['ap_lo']>= 90) & (df_1_IQR3['ap_lo'] < 100)), 'ap_bin'] = 2
df_1_IQR3.loc[(df_1_IQR3['ap_hi'] >= 160) | (df_1_IQR3['ap_lo'] >= 100), 'ap_bin'] = 3
## ap_hi 와 ap_lo 수치에 따라 범주화 하였음

print(df_1_IQR3)

df_1_IQR3 = df_1_IQR3.drop('ap_hi', axis = 1)
df_1_IQR3 = df_1_IQR3.drop('ap_lo', axis = 1)
## ap_hi, ap_lo 칼럼 삭제


def bar_chart(feature) :
    cardio_o = df_1_IQR3[df_1_IQR3['cardio'] == 1][feature].value_counts() 
    cardio_x = df_1_IQR3[df_1_IQR3['cardio'] == 0][feature].value_counts() 
    cardio = pd.DataFrame([cardio_o, cardio_x]) 
    cardio.index = ['cardio_o', 'cardio_x'] 
    cardio.plot(kind = 'bar', stacked = True, figsize = (10, 5)) 

# 성별에 따라서
bar_chart('gender')
plt.show()
## 비슷비슷하다.


# cholesterol에 따라서
bar_chart('cholesterol')
plt.show()
## 콜레스테롤이 높은사람이 걸리는 비율이 높다.

# gluc에 따라서
bar_chart('gluc')
plt.show()
## gluc(글루코스 수치)가 높은사람이 걸리는 비율이 높다.

# smoke에 따라서
bar_chart('smoke')
plt.show()
## 비슷비슷하다.


# alco에 따라서
bar_chart('alco')
plt.show()
## 비슷비슷하다.


# active에 따라서
bar_chart('active')
plt.show()
## 신체활동이 낮은사람이 걸리는 비율이 높다.(크게 높지는 않다.)


# age_bin에 따라서
bar_chart('age_bin')
plt.show()
## 나이가 많을수록 걸리는 비율이 높다.

# BMI_bin에 따라서
bar_chart('BMI_bin')
plt.show()
## BMI지수가 높을수록 걸리는 비율이 높다.

# ap_bin에 따라서
bar_chart('ap_bin')
plt.show()
## 혈압이 높을수록 걸리는 비율이 높다.

corr = df_1_IQR3.corr()
fig, ax = plt.subplots(figsize = (20, 10))
sns.heatmap(corr, annot = True)
plt.show()
## 상관계수 그래프 

train = df_1_IQR3.drop('cardio', axis = 1)
test = df_1_IQR3['cardio']

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, random_state =42)
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

print(gbc.score(X_test, y_test))

y_test_pred = gbc.predict(X_test)

from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion_matrix)

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_test_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)

plt.figure(figsize=(10,10))
plt.plot(false_positive_rate, true_positive_rate, color = 'red', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], linestyle = '--')
plt.show()

