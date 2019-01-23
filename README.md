# Stock-Prediction
Stock Prediction using Recurrent Neural Network
Pre-processing:
Pre-processing on the data (training and testing) includes simply using the closing value of the stocks for particular day for training and testing
Creation of Model:
A recurrent neural network is used to create the model to make predictions about next value of stocks price. The target output for any value of closing value is given by the subsequent value of the closing value (ie, the next day’s closing value)
Configuration of RNN :
1 layer of 4 LSTM nodes
1 output node
Number if epochs = 100
Batch size (number of values fed before backpropagation occurs) = 20
67% of the data is used for training and 33% is used for testing.
Prediction:
Prediction is done using the RNN created in the above step. A sequence of last 500 days is provided to the RNN. Its prediction is appended to the sequence of 500 values and this new sequence is passed to the RNN. This step is repeated 30 times to obtain the 30 values predicted by the RNN.

Code:
Predictor.py contains the code to predict the stocks for all companies and creates the submission file.
It first creates an RNN named ‘model’. The model is created using Sequential class from keras library. Transforms and inverse transforms are implemented using sklearn.preprocessing library.
