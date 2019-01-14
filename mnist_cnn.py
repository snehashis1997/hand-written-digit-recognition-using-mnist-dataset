
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

model=Sequential()
model.add(Convolution2D(64,4,4,input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

 
model.add(Flatten())
 
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
name='my_mnist_cnn.hdf5'
model.load_weights(name)

frame=cv2.imread('four.jpg')

roi=frame
roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
x1=20
y1=30
x2=600
y2=400

labels=['zero','one','two','three','four','five','six','seven','eight','nine']
font  = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 3
fontColor              = (0,0,255)
lineType  =2      
roi=roi/255
roi=cv2.resize(roi,(28,28))
roi=np.expand_dims(roi,axis=0)
roi=np.expand_dims(roi,axis=0)
roi=roi.reshape(1,28,28,1)
prob=model.predict(roi)
clas=model.predict_classes(roi)
clas=int(clas)
prob=prob.reshape(10,1)
text = labels[clas]
cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
cv2.imshow('frame',frame)
# When everything done, release the capture
cv2.waitKey(0)
cv2.destroyAllWindows()
