import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.optimizers import SGD 
from keras.layers import Dense , Conv2D , Flatten , MaxPooling2D
from numpy import mean, std

from sklearn.model_selection import KFold

# loading the dataset 
def load_dataset():
    (x_train , y_train ), (x_test,y_test)= mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0],28,28,1))
    x_test = x_test.reshape((x_test.shape[0],28,28,1))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def show(x_train):
    for i in range(1,10):
        plt.subplot(3,3,i)
        plt.imshow(x_train[i],plt.get_cmap("gray"))

    plt.show()

# Normalizing the pixel
def  norm_pixel(train,test):
    train_norm = train.astype(np.float32)
    test_norm = test.astype(np.float32)

    train_norm = train/255.0
    test_norm = test/255.0
    return train_norm, test_norm


# creating model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10 , activation = "softmax"))

    opt= SGD(learning_rate = 0.01 , momentum = 0.9)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Evaluating model
def eval_Model(X_data, Y_data, n_split=5):
    scores, histories = list(), list()
    kfold = KFold(n_splits=n_split, shuffle=True, random_state=1)
    
    # Iterate through the KFold splits
    for train_ind, test_ind in kfold.split(X_data):
        model = create_model()
        trainX, trainY, testX, testY = X_data[train_ind], Y_data[train_ind], X_data[test_ind], Y_data[test_ind]
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
    
    return scores, histories


# Ploting Graph
def summarize(histories):
    for i in range(len(histories)):
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()



# Ploting boxplot
def performance(scores):
 # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    plt.boxplot(scores)
    plt.show()



def run_model():
    trainX,trainY,testX,testY =load_dataset()
    trainX ,testX = norm_pixel(trainX,testX)
    score,histories= eval_Model(trainX,trainY)
    summarize(histories)
    performance(score)


# saving the final model
def save_model():
    trainX,trainY,testX,testY =load_dataset()
    trainX ,testX = norm_pixel(trainX,testX)
    
    model = create_model()
    model.fit(trainX,trainY,epochs=10,batch_size=32)
    model.save("MNIST_Model.keras")



save_model()