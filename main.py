import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import warnings

data =pd.read_csv('tablo/creditcard.csv')

data.head()
data.isnull().sum()
data.info()
data.describe().T.head()
data.shape
data.columns

dolandirma_vakalari=len(data[data['Class']==1])
print(' Dolandiricilik vaka sayisi:',dolandirma_vakalari)
non_dolandirma_vakalari=len(data[data['Class']==0])
print('Dolandiricilik disi vaka sayisi:',non_dolandirma_vakalari)
dolandirma=data[data['Class']==1]
gercek=data[data['Class']==0]
dolandirma.Amount.describe()
gercek.Amount.describe()
data.hist(figsize=(20,20),color='red')
plt.show()

rcParams['figure.figsize'] = 16, 8
f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Islem zamani ve sinifa gore tutar')
ax1.scatter(dolandirma.Time, dolandirma.Amount)
ax1.set_title('Dolandirma')
ax2.scatter(gercek.Time, gercek.Amount)
ax2.set_title('Gercek')
plt.xlabel('Zaman (saniyede)')
plt.ylabel('Miktar')
plt.show()

plt.figure(figsize=(10,8))
corr=data.corr()
sns.heatmap(corr,cmap='BuPu')
plt.show()

#Rasgele orman modeli
from sklearn.model_selection import train_test_split
X=data.drop(['Class'],axis=1)
y=data['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_st
ate=123)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
model=rfc.fit(X_train,y_train)
prediction=model.predict(X_test)
from sklearn.metrics import accuracy_score
sonuc=accuracy_score(y_test,prediction)*100
print("Rassal orman modeli doğruluk oranı:%"+str(sonuc))

#Lojistik Regrasyon modeli
from sklearn.linear_model import LogisticRegression
X1=data.drop(['Class'],axis=1)
y1=data['Class']
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3,rand
om_state=123)
lr=LogisticRegression()
model2=lr.fit(X1_train,y1_train)
prediction2=model2.predict(X1_test)
sonuc=accuracy_score(y1_test,prediction2)*100
print("Lojistik Regrasyon modeli doğruluk oranı:%"+str(sonuc))

#Karar Ağacı ile regrasyon
from sklearn.tree import DecisionTreeRegressor
X2=data.drop(['Class'],axis=1)
y2=data['Class']
dt=DecisionTreeRegressor()
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3,rand
om_state=123)
model3=dt.fit(X2_train,y2_train)
prediction3=model3.predict(X2_test)
sonuc=accuracy_score(y2_test,prediction3)*100
print("Lojistik Regrasyon modeli doğruluk oranı:%"+str(sonuc))
