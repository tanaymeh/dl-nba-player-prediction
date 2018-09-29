import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

data = pd.read_csv('nba ds.csv')

X = data.iloc[:,1:19]
Y = data.iloc[:,20]

trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.2,random_state=0)
testX = testX.fillna(testX.mean())
trainX = trainX.fillna(trainX.mean())
testY = testY.fillna(testY.mean())
trainY = trainY.fillna(trainY.mean())
sc = StandardScaler()
trainX = sc.fit_transform(trainX)
testX = sc.transform(testX)

model = Sequential()
model.add(Dense(units=10,kernel_initializer='uniform',activation='relu',input_dim=18))
model.add(Dense(units=10,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(trainX,trainY,batch_size=5,epochs=100)
predY = model.predict(testX)

predY = predY>0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testY,predY)