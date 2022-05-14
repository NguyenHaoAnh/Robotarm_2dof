#import numpy as np
import pandas as pd 
#import seaborn as sns 
import matplotlib.pyplot as plt 
#from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
#from keras.layers import Dense, Activation, Dropout, BatchNormalization, LSTM
from keras.models import Sequential
#from tensorflow.keras.utils import to_categorical
#from keras import callbacks
#from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score 
import csv
import math as m
l1 = 50
l2 = 40
with open('robot2.csv','w') as file:
  writer = csv.writer(file)
  writer.writerow(['tt1','tt2','Px','Py'])

  for tt1 in range(-180,181,1):
    for tt2 in range(-180,181,1):
      #tt1 = (tt1*m.pi)/180
      #tt2 = (tt2*m.pi)/180
      Px = l1*m.cos(tt1) + l2*m.cos(tt1+tt2)
      Py = l1*m.sin(tt1) + l2*m.sin(tt1+tt2)
      writer.writerow([tt1,tt2,Px,Py])
data=pd.read_csv('robot2.csv')
x=data.drop(data.columns[:2],axis=1)
y=data.drop(data.columns[2:],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout 
model=Sequential()
model.add(Dense(512,activation='relu',input_shape=(2,)))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.summary()
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
model.compile(loss='mse',optimizer=RMSprop(),metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size = 128,epochs=500,verbose=1,callbacks=[EarlyStopping(monitor='val_loss',patience=20)],validation_data = (x_test, y_test))
score=model.evaluate(x_test,y_test,verbose=0)
print('Sai số: ',score[0])
print('Độ chính xác: ',score[1])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(['train','Validation'])
plt.show()
model.save('robot2.h5')



