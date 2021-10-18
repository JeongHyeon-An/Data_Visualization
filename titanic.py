import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

## 데이터 불러오기
train = pd.read_csv('D:/pythonProject1/pofol/titanic/train.csv')
print(train.shape)
print(train.head(6))


## 탐험적 데이터 분석(EDA = Expoloratory Data Analysis)
# 데이터의 기본정보 확인하기
print(train.info())

# 데이터에서 비어있는 항목&수량 확인하기
print(train.isnull().sum()) #--> age: 177 / cabin: 684개가 isnull

# 데이터 요약
print(train.describe())


## Data Visualization
# 생존자 인원과 사망자 인원 확인하기
print(train.value_counts(['Survived'])) # ---> 생존자: 549 / 사망자: 342

# 생존여부에 따라 신규칼럼(Survived(humanized))에 Perish / Survived Value 삽입
train['Survived(humanized)'] = train['Survived'].replace(0,'Perish').replace(1,"Survived")

# 티켓 클래스에 따라 First class/ Business / Economy Value 삽입
train['Pclass(humanized)'] = train['Pclass'].replace(1,'First class')\
    .replace(2,'Business').replace(3,'Economy')
print(train.info())

# 티켓 클래스별 생존자수, 사망자수 그래프로 출력
print(train['Pclass(humanized)'].value_counts(['Survived']))
sns.countplot(x='Pclass(humanized)', hue='Survived(humanized)', palette='cool', data=train)
plt.show()


# 탑승지에 따라 신규칼럼(Embarked(humanized)) Chebourg / Southampton / Queenstown Value 삽입
train['Embarked(humanized)'] = train['Embarked'].replace(1,'Chebourg')\
    .replace(2,'Southampton').replace(3,'Queenstown')

# 탑승지별 생존자수, 사망자수 그래프로 출력
# print(train['Embarked(humanized)'].value_counts(['Survived']))
sns.countplot(x='Embarked(humanized)', hue='Survived(humanized)', data=train)
plt.show()


# 성별에 따른 생존자수, 사망자수 그래프로 출력
sns.countplot(x='Sex', hue='Survived(humanized)', data=train)
plt.show()