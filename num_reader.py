import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
mnist=keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#print(x_train[0])
#plt.imshow(x_train[0],cmap=plt.cm.binary)# cmap to binary (black/white)
#plt.show()

x_train=keras.utils.normalize(x_train,axis=1)
x_test=keras.utils.normalize(x_test,axis=1)

model=keras.models.Sequential()
model.add(keras.layers.Flatten())#input layer
model.add(keras.layers.Dense(128,activation=tf.nn.relu)) #nn.relu is default function ,hidden layer
model.add(keras.layers.Dense(128,activation=tf.nn.relu))#hidden layer
model.add(keras.layers.Dense(10,activation=tf.nn.softmax))#output layer

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)
val_loss,val_accuracy=model.evaluate(x_test,y_test)
print(val_loss)
print(val_accuracy)
model.save('num_reader.yml')
new_model=keras.models.load_model('num_reader.yml')
predictions=new_model.predict([x_test])
print(predictions[1])
print(np.argmax(predictions[1]))
plt.imshow(x_test[1])
plt.show()