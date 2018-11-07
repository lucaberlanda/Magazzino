import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import CoreFunctions.functions_involved as finv
import statsmodels as sm
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

sns.set_style('white')

m4_competition = True
moving_average = True

if m4_competition:

    """
    file_name = 'Monthly-train.csv'
    time_series_df = pd.read_csv(file_name,).T
    """

    for j in range(4, 6):

        time_series_df = finv.download_time_series_and_put_in_df([24600]).truncate(before='2010-12-30',
                                                                                   after=finv.get_last_friday())

        ts = time_series_df.iloc[2:, 0].pct_change().dropna()
        ts = ts.astype('float64')

        scaler = MinMaxScaler(feature_range=(0, 1))
        ts = scaler.fit_transform(ts.reshape(-1, 1))

        # split into train and test sets
        train_size = int(len(ts) * 0.90)
        test_size = len(ts) - train_size
        train, test = ts[0:train_size, :], ts[train_size:len(ts), :]

        # convert an array of values into a dataset matrix
        def create_dataset(dataset, look_back=100):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                b = dataset[i + look_back, 0]
                dataY.append(b)
            return np.array(dataX), np.array(dataY)

        # reshape into X=t and Y=t+1
        l_back = 1
        trainX, trainY = create_dataset(train, l_back)
        testX, testY = create_dataset(test, l_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, l_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=j, batch_size=1, verbose=2)

        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(ts)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[l_back:len(trainPredict) + l_back, :] = trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(ts)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (l_back* 2) + 1:len(ts) - 1, :] = testPredict

        # plot baseline and predictions
        plt.plot(scaler.inverse_transform(ts))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()

        aaa = pd.Series(testPredictPlot.flatten()).dropna()
        aaa.iloc[0] = 0
        # aaa += 1
        # aaa = aaa.cumprod()
        bbb = pd.Series(scaler.inverse_transform(ts).flatten()).loc[aaa.index[0]:aaa.index[-1]]
        bbb.iloc[0] = 0
        # bbb += 1
        # bbb = bbb.cumprod()
        # pd.concat([aaa, bbb], axis=1).plot()
        # plt.scatter(aaa, bbb)

        ret_f_rolling_std = aaa.rolling(window=20).std()
        ret_and_std = pd.concat([aaa, 1 * ret_f_rolling_std], axis=1)
        ret_and_std['position_long'] = ret_and_std.loc[:, 0] > ret_and_std.loc[:, 1]
        ret_and_std['position_short'] = -ret_and_std.loc[:, 0] > ret_and_std.loc[:, 1]
        ret_and_std.loc[ret_and_std['position_long'] == True, 'position'] = 1.5
        ret_and_std.loc[ret_and_std['position_long'] == False, 'position'] = 1
        ret_and_std.loc[ret_and_std['position_short'] == True, 'position'] = 0.5
        conv_aux = pd.concat([bbb, ret_and_std.loc[:, 'position']], axis=1)

        conv_aux.iloc[:, 1].plot()
        plt.show()

        position_w_conviction = conv_aux.loc[:, 0] * conv_aux.loc[:, 'position']
        position_w_conviction += 1
        position_w_conviction = position_w_conviction.cumprod()
        bbb += 1
        bbb = bbb.cumprod()
        to_plot = pd.concat([position_w_conviction, bbb], axis=1)
        to_plot.columns = ['with_conviction', 'original_ts']
        to_plot.plot()
        plt.show()

else:
    time_series_df = finv.download_time_series_and_put_in_df([24600, 29037, 32718, 14479, 14566, 18871, 103167]).\
        truncate(before='2010-12-30', after=finv.get_last_friday())


def plot_acf(s):
    sm.graphics.tsaplots.plot_acf(s, lags=50)
    sm.graphics.tsaplots.plot_pacf(s, lags=50)
    plt.show()