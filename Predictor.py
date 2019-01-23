import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
numpy.random.seed(7)
dataframe = read_csv('titan1.csv', usecols=[1], engine='python')
dataframe = dataframe.dropna()
dataset = dataframe.values
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=20)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

newdataset=dataset[len(dataset)-500:,:]

newplot = numpy.empty([30,1])
newx,newy = create_dataset(newdataset,1)
newx=numpy.reshape(newx,(newx.shape[0],1,newx.shape[1]))
for i in range(30):
	newpredict=model.predict(newx)
	newplot[i] = newpredict[len(newpredict)-1]
	rowtoadd=numpy.reshape(newpredict[len(newpredict)-1],(newpredict[len(newpredict)-1].shape[0],1,1))

	newx=numpy.vstack((newx,rowtoadd))
	
newplot = scaler.inverse_transform(newplot)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

newplotg=numpy.empty([len(dataset)+30,1])
newplotg[:,:] = numpy.nan
for i in range(len(dataset),len(dataset)+30):
	newplotg[i]=newplot[i-len(dataset)]

dataset=scaler.inverse_transform(dataset)
print(dataset[len(dataset)-1])
print(newplotg[len(newplotg)-1])
print((newplotg[len(newplotg)-1]-dataset[len(dataset)-1])/dataset[len(dataset)-1])
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(newplotg)
plt.show()