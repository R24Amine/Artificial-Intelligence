import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D
from keras.models import Model
from keras.layers import Input 
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

def get_data_set(fpath):
    imgs=[]
    labels=[]
    print('readin files..')
    for f in os.listdir(fpath):
        if not (f.endswith('pgm')):
            labels.append(f.split('.')[0])
            print('readin file '+f)
            img=np.asarray(Image.open(fpath+f))
            imgs.append(img)
    print('reading files finished')
    return np.asarray(imgs),labels

fpath=('C:/Users/PC_AMINE/Desktop/yalefaces')
imgs,labels=get_data_set(fpath)

labels1=[labels[i][-2:]for i in range(len(labels))]

xtrain,xtest,ytrain,ytest=train_test_split(imgs,labels1,test_size=0.33)
xtrain1=xtrain.reshape(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2],1)
xtest1=xtest.reshape(xtest.shape[0],xtest.shape[1],xtest.shape[2],1)

ytrain1=to_categorical(ytrain)
ytest1=to_categorical(ytest)

ytrain2=ytrain1[:,1:]
ytest2=ytest1[:,1:]

'''visible=Input(shape=(4,))
hidden=Dense(4)(visible)
clf=Model(inputs=visible,outputs=hidden)'''

from keras.models import Sequential
clf=Sequential()

clf.add(Convolution2D(32,kernel_size=9,input_shape=(243,320,1),activation='relu'))
clf.add(MaxPooling2D(pool_size=(8,8)))
clf.add(Convolution2D(32,kernel_size=3,activation='relu'))
clf.add(Convolution2D(32,kernel_size=3,activation='relu'))
clf.add(Convolution2D(16,kernel_size=3,activation='relu'))
clf.add(MaxPooling2D(pool_size=(4,4)))

clf.add(Flatten())
clf.add(Dense(activation='relu',units=248))
clf.add(Dense(activation='relu',units=128))
clf.add(Dense(activation='softmax',units=15))

#compiling the cnn
clf.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
clf.fit(xtrain1,ytrain2,validation_data=(xtest1,ytest2),epochs=50,batch_size=32)