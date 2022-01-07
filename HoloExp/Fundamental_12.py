import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


print(sklearn.__version__) # 사이킷런 버전확인

r = np.random.RandomState(10) # 난수를 생성
x = 10 * r.random(100)
y = 2 * x - 3 * r.rand(100) # 그래프 생성

# 생성된 산점도 시각화
plt.scatter(x, y)
plt.show()

print(x.shape) # 1차원 벡터 x 표시
print(y.shape) # 1차원 벡터 y 표시

model = LinearRegression() # LinearRegression 모델 생성
print(model)

X = x.reshape(100, 1) # x를 특성행렬로 재구성

model.fit(X, y) # 모델 훈련

x_new = np.linspace(-1, 11, 100) # linespace생성 ( start, stop, num(갯수))
X_new = x_new.reshape(100, 1) # 특성행렬으로 재구성
y_new = model.predict(X_new) # X_new 예측

error = np.sqrt(mean_squared_error(y, y_new)) # RMSE성능평가
print(error)

# 라인 생성
plt.scatter(x, y, label = 'input data')
plt.plot(X_new, y_new, color = 'red', label = 'regression line')
plt.show()

print('----------------------------------')

from sklearn.datasets import load_wine

data = load_wine() # 와인 데이터 생성

print(data.keys()) # 데이터의 키값

print(data.data.shape) # 데이터의 형태
print(data.data.ndim) # 데이터의 차원
print(data.target.shape) # 타겟의 형태
print(data.feature_names) # 특성들의 이름
print(data.target_names) # 타겟들의 이름

print(data.DESCR)# 위의 모든 정보를 포함한 여러 정보들 (사실 이것만 있으면 위에꺼 다 있다....)

print('----------------------------------')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

P = pd. DataFrame(data.data, columns=data.feature_names) # 데이터 프레임 생성 (데이터, 열(속성) 이름)

X = data.data # 특성행렬
y = data.target # 타겟행렬

model = RandomForestClassifier() # 모델 생성

model.fit(X, y) # 훈련

y_pred = model.predict(X) # 예측

print(classification_report(y, y_pred)) # 타겟 벡터 즉 라벨인 변수명 y와 예측값 y_pred을 각각 인자로 넣습니다.
print("accuracy = ", accuracy_score(y, y_pred)) # 정확도를 출력합니다.

print('----------------------------------')

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_wine() # 데이터 생성

X_train = data.data[:142] # 178개의 데이터중 142개를 할당
X_test = data.data[142:] # 178개의 데이터중 36(178-142)개를 할당
y_train = data.target[:142]
y_test = data.target[142:]

model = RandomForestClassifier() #모델 생성
model.fit(X_train, y_train) # 훈련
y_pred = model.predict(X_test) # 예측
print("정답률=", accuracy_score(y_test, y_pred)) # 정답률 확인

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# 위의 데이터를 나누는 과정을 함수를 사용해서 수행(특성행렬, 타겟행렬, 나누는 비율, 랜덤 시드)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)