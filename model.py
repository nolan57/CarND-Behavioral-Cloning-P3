
# coding: utf-8

# In[1]:

import csv
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard,EarlyStopping,CSVLogger
from keras.models import Sequential
from keras.layers import Input,Lambda
from keras.layers.noise import GaussianNoise
from keras.layers.core import Dense,Activation,Flatten,Dropout
from keras.layers.convolutional import Conv2D,Cropping2D
from keras.layers.pooling import MaxPool2D


# In[2]:

import sklearn

records=[]
paths=[]
imgs=[]
angels=[]
adj=0.2

with open('./data/driving_log.csv') as csvf:
    cr=csv.DictReader(csvf,delimiter=',')
    for record in cr:
        records.append(record)
#print(len(records))

def generator(records,batch_size=32):
    total=len(records)
    root='./data/'
    while 1:
        sklearn.utils.shuffle(records)            
        for offset in range(0,total,batch_size):
            batch_record=records[offset:offset+batch_size]
            imgs=[]
            angels=[]
            for record in batch_record:
                csteering=float(record['steering'])

                if(csteering>0.5 or csteering<-0.5):
                    print('pass!')
                    continue
 
                lsteering=csteering+adj
                rsteering=csteering-adj
                
                path=root+record['center'].replace(" ","")
                img=cv2.imread(path)
                if(img is not None):
                    imgs.append(img)
                    angels.append(csteering)
                    img=np.fliplr(img)
                    imgs.append(img)
                    angels.append(-csteering)
        
                path=root+record['left'].replace(" ","")
                img=cv2.imread(path)
                if(img is not None):
                    imgs.append(img)
                    angels.append(lsteering)
                    img=np.fliplr(img)
                    imgs.append(img)
                    angels.append(-lsteering)
        
                path=root+record['right'].replace(" ","")
                img=cv2.imread(path)
                if(img is not None):
                    imgs.append(img)            
                    angels.append(rsteering)
                    img=np.fliplr(img)
                    imgs.append(img)
                    angels.append(-rsteering)
            
            batch_x=np.array(imgs)
            batch_y=np.array(angels)
            yield sklearn.utils.shuffle(batch_x, batch_y)


# In[3]:

from sklearn.model_selection import train_test_split
train_example,valid_example=train_test_split(records,test_size=0.2)
train_generator=generator(train_example,batch_size=32)
valid_generator=generator(valid_example,batch_size=32)


# In[5]:

model=Sequential()
model.add(Lambda(lambda x:x/127.5-1.,
                 input_shape=(160,320,3),
                 output_shape=(160,320,3)))
model.add(GaussianNoise(stddev-1.))
model.add(Cropping2D(cropping=((80,30),(20,20)),input_shape=(160,320,3)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(400))
model.add(Dense(100))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
tb=TensorBoard(log_dir='./logs',histogram_freq=1,write_images=True)
es=EarlyStopping(monitor='val_loss',min_delta=0.002,patience=3)
cl=CSVLogger(filename='t.log',separator=',')


# In[ ]:

steps_per_epoch=len(train_example)
nb_val_samples=len(valid_example)
history=model.fit_generator(train_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=valid_generator,
                            nb_val_samples=nb_val_samples,
                            nb_epoch=10,
                            callbacks=[tb,es,cl])
model.save('./model.h5')


# In[ ]:

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model

#plot_model(model,to_file='no_more_dense_model.png')
#SVG(model_to_dot(model).create(prog='dot',format='svg'))


# In[1]:

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()


# In[ ]:



