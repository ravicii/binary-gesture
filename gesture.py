import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
'''

vid=cv.VideoCapture(0)
x_train,y_train=[],[]
print(1,'training')
while(True):
    ret,frame=vid.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('th-up',frame)
    frame=cv.resize(frame,(150,150))
    frame=frame.reshape(150,150,1)
    x_train.append(frame)
    y_train.append(1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()
vid=cv.VideoCapture(0)
print(0,'training')
while(True):
    ret,frame=vid.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('th-dn',frame)
    frame=cv.resize(frame,(150,150))
    frame=frame.reshape(150,150,1)
    x_train.append(frame)
    y_train.append(0)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()
x_train=np.array(x_train)
y_train=np.array(y_train)
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),input_shape=(150,150,1),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
train_datagen=ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
i=int(input('Enter a number b/w 0-1000'))
plt.imshow(x_train[i],cmap=plt.cm.binary)
maping=['THUMBS DOWN','THUMBS UP']
print(maping[y_train[i]])
plt.show()
model.fit(
    train_datagen.flow(
        x_train,
        y_train,
        batch_size=20,
        shuffle=True
    ),
    epochs=5,
    verbose=1,
)
model.save('gesture_model.h5')
'''
model=tf.keras.models.load_model('gesture_model.h5')
maping=['THUMBS DOWN','THUMBS UP']
vid=cv.VideoCapture(0)
while(True):
    ret,frame=vid.read()
    cv.imshow('capture',frame)
    frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame=cv.resize(frame,(150,150))
    frame=frame.reshape(1,150,150,1)
    pred=model.predict(frame)
    if pred[0]>0:
        print(maping[1])
    else:
        print(maping[0])
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()