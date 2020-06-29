from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.sql.functions import col
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants
from PredictionAlgorithms.PredictiveRegression.PredictiveRandomForestRegression import PredictiveRandomForestRegression
from PredictionAlgorithms.PredictiveRegression.PredictiveRegression import PredictiveRegression
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pUtil
from pyspark.ml.evaluation import RegressionEvaluator


class PredictiveGradientBoostRegression(PredictiveRegression):

    def __init__(self):
        pass

    def gradientBoostRegression(self, regressionInfo):
        spark = ''  # temp fix
        etlStats = self.etlOperation(etlInfo=regressionInfo)

        featuresColm = etlStats.get(PredictiveConstants.FEATURESCOLM)
        modelName = regressionInfo.get(PredictiveConstants.MODELSHEETNAME)
        labelColm = etlStats.get(PredictiveConstants.LABELCOLM)
        trainData = etlStats.get(PredictiveConstants.TRAINDATA)
        algoName = regressionInfo.get(PredictiveConstants.ALGORITHMNAME)
        locationAddress = regressionInfo.get(PredictiveConstants.LOCATIONADDRESS)
        modelId = regressionInfo.get(PredictiveConstants.MODELID)

        gradientBoostRegressorModelFit = \
            GBTRegressor(labelCol=labelColm,
                         featuresCol=featuresColm,
                         predictionCol=modelName)
        regressor = gradientBoostRegressorModelFit.fit(trainData)
        # predictionData = regressor.transform(self.testData)

        regressionStat = self.regressionEvaluation(regressor=regressor, regressionInfo=regressionInfo,
                                                   etlStats=etlStats)

        # persisting model
        modelNameLocal = "randomForestModel"
        extention = ".parquet"
        modelStorageLocation = locationAddress + modelId.upper() + modelNameLocal.upper() + extention
        regressor.write().overwrite().save(modelStorageLocation)

        regressionStat["modelPersistLocation"] = {"modelName": modelNameLocal,
                                                  "modelStorageLocation": modelStorageLocation}

        return regressionStat

    def createGraphData(self, regressor, regressionInfo, etlStats):
        pass

    def regressionEvaluation(self, regressor, regressionInfo, etlStats):
        # using the randomForestCode.
        predictiveRandomForestRegression = PredictiveRandomForestRegression()
        response = predictiveRandomForestRegression.regressionEvaluation(regressor, regressionInfo, etlStats)

        return response
