import math
import re
import time

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pmdarima import auto_arima
from pyspark.sql import SparkSession
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants
#
# spark = \
#     SparkSession.builder.appName('predictive_Analysis').master('local[*]').getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')


class ForecastingModel:
    global spark
    def __init__(self, alpha, beta, gamma, isTrending,
                 isSeasonal, seasonalPeriodsManual,
                 seasonalP, seasonalD, seasonalQ, confIntPara,sparkSession):
        global spark
        spark = sparkSession
        self.alpha = None if alpha == None else float(alpha)
        self.beta = None if beta == None else float(beta)
        self.gamma = None if gamma == None else float(gamma)
        self.seasonalPeriodsManual = int(seasonalPeriodsManual)
        self.isTrending = True if isTrending == "true" else None
        self.isSeasonal = True if isSeasonal == "true" else False
        self.seasonalP = 0 if seasonalP == None else int(seasonalP)
        self.seasonalD = 0 if seasonalD == None else int(seasonalD)
        self.seasonalQ = 0 if seasonalQ == None else int(seasonalQ)
        self.confIntPara = 0.05 if confIntPara == None else round(1 - float(confIntPara), 2)

    def forecastingTimeSeries(self, data, count, len_type, model_type, trendType, seasonType, forecastAlgorithm, P, Q,
                              D,
                              arima_model_type, iterations,
                              modelName, locationAddress, columnsNameList,
                              sheetId):
        start_time = time.time()
        self.model_type = model_type
        arima_model_type = None if arima_model_type == None else arima_model_type
        self.P = 0 if P == None else int(P)
        self.D = 0 if D == None else int(D)
        self.Q = 0 if Q == None else int(Q)
        self.trendType = None if trendType == None else trendType
        self.seasonType = None if seasonType == None else seasonType
        self.maxIterations = None if iterations == None else int(iterations)
        self.trendPara = "ct" if self.isTrending == True else None
        df = pd.DataFrame(data)
        df = df[df[len(df.columns) - 1] != 'null']
        df[df.columns[-1]] = pd.to_datetime(df[df.columns[-1]], infer_datetime_format=True)
        newdates = []
        calculatedPDQ = []
        if len_type == 'SecondaryYear':
            frequency = "AS"
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(years=1), periods=count, freq='AS')
        if len_type == 'SecondaryMonth':
            frequency = "MS"
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(months=1), periods=count, freq='MS')
        if len_type == 'SecondaryQuarter':
            frequency = "QS"
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(months=3), periods=count, freq='QS')
        if len_type == 'SecondaryWeekNumber':
            frequency = "W"
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(weeks=1), periods=count, freq='W')
        if len_type == 'SecondaryDay':
            frequency = "D"
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(days=1), periods=count, freq='D')
        # for future use
        if len_type == 'SecondaryHour':
            frequency = "H"
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(hour=1), periods=count, freq='H')
        if len_type == 'SecondaryMinute':
            frequency = "T"
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(minute=1), periods=count, freq='T')
        if len_type == 'SecondarySecond':
            frequency = "S"
            newdates = pd.date_range(df.iloc[-1, -1] + relativedelta(second=1), periods=count, freq='S')

        newdatesDataframe = pd.DataFrame(newdates.values, columns=["dateColm"])
        newdatesDataframe["dateColm"] = pd.to_datetime(newdatesDataframe["dateColm"], infer_datetime_format=True)
        newdatesDataframe = newdatesDataframe.set_index("dateColm")
        newdatesDataframe.index.freq = frequency
        self.newdatesDataframe = newdatesDataframe
        self.frequency = frequency
        newdates = pd.to_datetime(newdates)
        newdates = pd.DataFrame(newdates)
        self.newdates = newdates
        dfTrans = pd.DataFrame(data)
        columnNumbers = len(dfTrans.columns)
        forecastedResultList = []
        confidenceIntervalList = []
        summaryStatsObject = {}

        newDatesList = []
        for date in newdates[0].tolist():
            newDatesList.append(str(date))

        endDate = newDatesList[-1]
        startDate = newDatesList[0]
        # create an object to store multiple parquet dataset info
        parquetDatasetInfo = {}
        for x in range(0, columnNumbers - 1):
            dfTransData = dfTrans.iloc[:, [x, -1]].values
            dfColmHolt = pd.DataFrame(dfTransData, columns=['target', 'indexed'])
            dfColmHolt['indexed'] = pd.to_datetime(dfColmHolt['indexed'], infer_datetime_format=True)
            dfColmIndHOlt = dfColmHolt.set_index('indexed')
            dfColmIndHOlt = dfColmIndHOlt.astype(np.float)
            dfColmIndHolt = dfColmIndHOlt.dropna()
            self.dfColmIndHolt = dfColmIndHolt.asfreq(frequency).fillna(0)
            if forecastAlgorithm == "holtw" or forecastAlgorithm == "holtW":
                if model_type == 'Automatic':
                    result = self.holtWintersForecasting()
                if model_type == "Custom":
                    result = self.holtWintersForecasting()
            if forecastAlgorithm == "arima":
                if arima_model_type == "arimaAutomatic":
                    result = self.autoArimaForecasting()
                if arima_model_type == "arimaCustom":
                    result = self.seasonalArimaForecasting()
            forecastedResultList.append(result)
            # writing data to parquet
            # do the refactoring of this code at the end
            dateColumnSpecialChar = re.sub('[^a-zA-Z0-9]', '_',columnsNameList[-1])  #call the relaceSpecialCharacter from predictiveUtilities
            
            fieldColumnSpecialChar = re.sub('[^a-zA-Z0-9]', '_',columnsNameList[x])

            datesLength = len(newDatesList)
            indexRange = list(range(0, datesLength))
            indexColmName = PredictiveConstants.DMXINDEX

            tempDatasetDict = {indexColmName:indexRange,
                               dateColumnSpecialChar: newDatesList,
                               fieldColumnSpecialChar: result}
            tempDataset = pd.DataFrame.from_dict(tempDatasetDict)
            tempDatasetSpark = spark.createDataFrame(tempDataset)
            modelNameWithField = modelName + columnsNameList[x]
            # newstring = re.sub('[^a-zA-Z0-9]', '', string)

            # predictiveUtilitiesObj = PredictiveUtilities()
            datasetInfo = \
                self.writeToParquet(fileName=modelNameWithField,
                                    locationAddress=locationAddress,
                                    userId=sheetId,
                                    data=tempDatasetSpark)
            parquetDatasetInfo[columnsNameList[x]] = datasetInfo

            summaryStatsObject[x] = self.summaryObj

            confidenceIntervalList.append(self.confidenceInterval)

        # creating a dataframe of the forecasted datamakeDatasetDict = {}
        # makeDatasetDict = {}
        # for colms,name in zip(forecastedResultList,columnsNameList):
        #     makeDatasetDict[name] = colms  #forecastedResultList.index(colms)
        # makeDatasetDict["TimeStamp"] = newDatesList
        # finalDataset = pd.DataFrame.from_dict(makeDatasetDict) #columns=["colm1","colm2"]
        # finalParquetDataset = spark.createDataFrame(finalDataset)

        # calling write to prquet method from predictive utilities class
        # predictiveUtilitiesObj = PredictiveUtilities()
        # finalDatasetInfo = \
        #     predictiveUtilitiesObj.writeToParquet(fileName=modelName,
        #                                           locationAddress=locationAddress,
        #                                           userId=sheetId,
        #                                           data=finalParquetDataset)

        # add startDate,endDate,fields name,parquetName....
        json_response = {'run_status': 'success',
                         'arimaParams': calculatedPDQ,
                         'execution_time': time.time() - start_time,
                         "summaryObj": summaryStatsObject,
                         "confidenceInterval": confidenceIntervalList,
                         "parquetDatasetInfo": parquetDatasetInfo,
                         "startDate": startDate,
                         "endDate": endDate
                         }
        # return str(json.dumps(json_response)).encode('utf-8')
        return json_response
    def holtWintersForecasting(self):
        shouldDampTrendComponent = self.trendType in ['additive', 'multiplicative'] and self.seasonType is None
        if self.trendType != "multiplicative":
            self.trendType = "additive" if self.isTrending == True else None
        if self.seasonType != "multiplicative":
            self.seasonType = "additive" if self.isSeasonal == True else None

        try:
            HoltWinter = ExponentialSmoothing(self.dfColmIndHolt, seasonal_periods=self.seasonalPeriodsManual,
                                              trend=self.trendType,
                                              seasonal=self.seasonType).fit(smoothing_level=self.alpha,
                                                                            smoothing_slope=self.beta,
                                                                            smoothing_seasonal=self.gamma,
                                                                            optimized=True)
        except:
            if self.dfColmIndHolt.shape[0] <= 12 and self.frequency == "MS":
                self.seasonalPeriodsManual = 2
            HoltWinter = ExponentialSmoothing(self.dfColmIndHolt, seasonal_periods=self.seasonalPeriodsManual,
                                              trend=self.trendType,
                                              seasonal=self.seasonType).fit(smoothing_level=self.alpha,
                                                                            smoothing_slope=self.beta,
                                                                            smoothing_seasonal=self.gamma,
                                                                            optimized=True)
        alphaStats = HoltWinter.params["smoothing_level"]
        betaStats = HoltWinter.params["smoothing_slope"]
        gammaStats = HoltWinter.params["smoothing_seasonal"]
        AICValue = HoltWinter.aic
        BICValue = HoltWinter.bic
        AICCValue = HoltWinter.aicc
        residualsValue = HoltWinter.resid
        actualValues = list(HoltWinter.data.endog)
        SSE = HoltWinter.sse
        countForRows = self.dfColmIndHolt["target"].count()
        # RMSE = math.sqrt(HoltWinter.sse / countForRows)
        fittedValues = list(HoltWinter.fittedvalues)
        RMSE = math.sqrt(mean_squared_error(actualValues, fittedValues))
        rSquare = r2_score(actualValues, fittedValues)
        self.summaryObj = {"AICValue": AICValue, "BICValue": BICValue, "AICCValue": AICCValue,
                           "seasonalPeriodsValue": self.seasonalPeriodsManual,
                           "isTrend": self.isTrending, "isSeasonal": self.isSeasonal, "alphaStats": alphaStats,
                           "betaStats": betaStats,
                           "gammaStats": gammaStats}
        prediction = HoltWinter.predict(start=self.newdatesDataframe.index[0], end=self.newdatesDataframe.index[-1])
        prediction = list(prediction.values)
        self.confidenceInterval = [[]]

        # for future use
        '''
        import statistics as stat
        # print(fittedValues)
        countForRows = self.dfColmIndHolt["target"].count()
        fittedValuesMean=stat.mean(fittedValues)
        fittedValuesStd=stat.stdev(fittedValues)
        print(fittedValuesMean,fittedValuesStd)
        confIntUpper=[]
        confIntLower=[]
        for forecastedVal in fittedValues:
            confIntUpper.append(forecastedVal+((1.96*fittedValuesStd)/math.sqrt(countForRows)))
            confIntLower.append(forecastedVal-((1.96*fittedValuesStd)/math.sqrt(countForRows)))
            print(forecastedVal-((1.96*fittedValuesStd)/math.sqrt(countForRows)),forecastedVal,forecastedVal+((1.96*fittedValuesStd)/math.sqrt(countForRows)))
        # print(confIntLower,confIntUpper)
        '''

        return prediction

    # manual
    def seasonalArimaForecasting(self):
        self.seasonalOrderPara = (self.seasonalP, self.seasonalD, self.seasonalQ, self.seasonalPeriodsManual)
        self.orderPara = (self.P, self.D, self.Q)
        if self.isSeasonal == False:
            self.seasonalOrderPara = (0, 0, 0, 1)
            try:
                model = SARIMAX(endog=self.dfColmIndHolt, order=self.orderPara, seasonal_order=self.seasonalOrderPara,
                                trend=self.trendPara)
                modelFit = model.fit(disp=0)
            except:
                model = SARIMAX(endog=self.dfColmIndHolt, order=self.orderPara,
                                trend=self.trendPara)
                modelFit = model.fit(disp=0)
        else:
            try:
                model = SARIMAX(endog=self.dfColmIndHolt, order=self.orderPara, seasonal_order=self.seasonalOrderPara,
                                trend=self.trendPara)
                modelFit = model.fit(disp=0)
            except Exception as e:
                print(str(e))
                # self.seasonalOrderPara = (0, 0, 0, self.seasonalPeriodsManual)
                # model = SARIMAX(endog=self.dfColmIndHolt, order=self.orderPara, seasonal_order=self.seasonalOrderPara,
                #                 trend=self.trendPara)
                # modelFit = model.fit(disp=0)
        AICValue = modelFit.aic
        BICValue = modelFit.bic
        HQICValue = modelFit.hqic
        orderValue = model.order
        seasonalOrderValue = model.seasonal_order
        seasonalPeriodsValue = model.seasonal_periods
        trendValue = model.trend
        actualValues = list(modelFit.data.endog)
        fittedValues = list(modelFit.fittedvalues)
        RMSE = math.sqrt(mean_squared_error(actualValues, fittedValues))
        rSquare = r2_score(actualValues, fittedValues)
        orderParam = ('p', 'd', 'q')
        orderParamDict = {}
        for val, para in zip(orderValue, orderParam):
            orderParamDict[para] = val
        seasonalOrderParam = ("P", "D", "Q")
        seasonalOrderParamDict = {}
        for val, para in zip(seasonalOrderValue, seasonalOrderParam):
            seasonalOrderParamDict[para] = val

        predictionModel = modelFit.get_prediction(self.newdatesDataframe.index[0], self.newdatesDataframe.index[-1])
        self.confidenceInterval = (predictionModel.conf_int(alpha=self.confIntPara)).values.tolist()
        prediction = list(predictionModel.predicted_mean.values)
        self.summaryObj = {"AICValue": AICValue, "BICValue": BICValue, "HQICValue": HQICValue,
                           "trendValue": trendValue, "seasonalPeriodsValue": seasonalPeriodsValue,
                           "isTrend": self.isTrending, "isSeasonal": self.isSeasonal,
                           "orderParamDict": orderParamDict, "seasonalOrderParamDict": seasonalOrderParamDict
                           }

        '''
        calculating the std errors and conf from forecast option
        error=mean_squared_error(realData,predictedData)-> using the sklearn library
        forecast, stderr, conf=modelFit.forecast(alpha=0.95)
        '''
        # forecastedValues = modelFit.forecast(len(newdates)).values
        return prediction

    # automatic
    def autoArimaForecasting(self):
        try:
            modelAuto = auto_arima(self.dfColmIndHolt, m=self.seasonalPeriodsManual, max_p=5, max_q=5, trace=True,
                                   error_action="ignore", stepwise=True, max_order=25, transparams=True,
                                   suppress_warnings=True,
                                   maxiter=int(self.maxIterations), seasonal=self.isSeasonal, trend=self.trendPara,
                                   with_intercept=self.isTrending)
        except Exception as e:
            print("exception :" + str(e))
            if type(e) is ValueError and str(e).startswith("Found array with 0 sample"):
                self.seasonalPeriodsManual = 1
            if self.dfColmIndHolt.shape[0] <= 12 and self.frequency == "MS":
                self.seasonalPeriodsManual = 1
            modelAuto = auto_arima(self.dfColmIndHolt, m=self.seasonalPeriodsManual, max_p=3, max_q=3, trace=True,
                                   error_action="ignore",
                                   suppress_warnings=True, stepwise=True, max_order=10, maxiter=int(self.maxIterations),
                                   seasonal=self.isSeasonal)
        AICValue = modelAuto.arima_res_.aic
        BICValue = modelAuto.arima_res_.aic
        HQICValue = modelAuto.arima_res_.aic
        orderValue = modelAuto.order
        seasonalOrderValue = modelAuto.seasonal_order
        trendValue = modelAuto.trend
        actualValues = list(modelAuto.arima_res_.data.endog)
        fittedValues = list(modelAuto.arima_res_.fittedvalues)
        RMSE = math.sqrt(mean_squared_error(actualValues, fittedValues))
        rSquare = r2_score(actualValues, fittedValues)
        orderParam = ('p', 'd', 'q')
        orderParamDict = {}
        for val, para in zip(orderValue, orderParam):
            orderParamDict[para] = val
        seasonalOrderParam = ("P", "D", "Q", "seasonalPeriods")
        seasonalOrderParamDict = {}
        for val, para in zip(seasonalOrderValue, seasonalOrderParam):
            seasonalOrderParamDict[para] = val
        self.summaryObj = {"AICValue": AICValue, "BICValue": BICValue, "HQICValue": HQICValue,
                           "trendValue": trendValue,
                           "isTrend": self.isTrending, "isSeasonal": self.isSeasonal,
                           "orderParamDict": orderParamDict, "seasonalOrderParamDict": seasonalOrderParamDict}
        prediction, confint = modelAuto.predict(n_periods=len(self.newdates), return_conf_int=True,
                                                alpha=self.confIntPara)
        self.confidenceInterval = confint.tolist()
        prediction = list(prediction)

        return prediction

    # overwriting this method since it is not yet linked with the predictive analysis.
    # once it got linked this method is written in predictiveUtilities.
    def writeToParquet(self, fileName, locationAddress, userId, data):
        extention = ".parquet"
        fileName = fileName.upper()
        userId = userId.upper()
        fileNameWithPath = locationAddress + userId + fileName + extention
        data.write.parquet(fileNameWithPath, mode="overwrite")
        onlyFileName = userId + fileName
        result = {"fileNameWithPath": fileNameWithPath,
                  "onlyFileName": onlyFileName}
        return result
