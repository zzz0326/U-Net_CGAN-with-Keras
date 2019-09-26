#first LSTM

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
def LSTM_model():
    # we use 20 data as the imput
    dataX=[]
    for i in range(1,11):
        data = []
        for j in range(0,20):
            data.append(j*2+i)
            print(data)
        dataX.append([data])
    #dataX = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #output
    dataY = [1,0,1,0,1,0,1,0,1,0]
    #length for each cell ===>20
    length = 20
    #reshape X to be [samples,time steps,feature]
    #regred 20 datas as the different features
    X = numpy.reshape(dataX,(10,length,1))
    #A归一化处理 输入需要的是0-1之间的数据
    X = X/float(len(dataX))
    y = np_utils.to_categorical(dataY)

    #构建模型
    model = Sequential()
    #input
    model.add(LSTM(2,input_shape=(X.shape[1],X.shape[2])))
    #一个LSTM当中有32个cell
    #output
    model.add(Dense(y.shape[1], activation='softmax'))
    #全连接层 然后softmax 进行分类学习
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, nb_epoch=500, batch_size=1, verbose=2)
    #完成构建

    scores = model.evaluate(X, y, verbose=0)
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))

    for pattern in dataX:
        x = numpy.reshape(pattern, (1,20, 1))
        x = x / float(10)
        prediction = model.predict(x, verbose=0)
        print(prediction)
        index = numpy.argmax(prediction)

        print (index)

LSTM_model()
#如果输入的长度是会改变的话 可以使用pad_sequences()函数 对输入进行0的填充 time_step 不需要改变 只需要维持max_len