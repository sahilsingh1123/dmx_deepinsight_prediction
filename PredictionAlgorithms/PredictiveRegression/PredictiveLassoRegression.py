from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants
from PredictionAlgorithms.PredictiveRegression.PredictiveLinearRegression import PredictiveLinearRegression
from PredictionAlgorithms.PredictiveRegression.PredictiveRegression import PredictiveRegression
from pyspark.sql.types import *
from pyspark.sql.functions import col
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pUtil


class PredictiveLassoRegression(PredictiveRegression):

    def __init__(self):
        pass

    def lassoRegression(self, regressionInfo):
        # calling etl method for etl Operation
        regParam = 0.05
        etlStats = self.etlOperation(etlInfo=regressionInfo)

        featuresColm = etlStats.get(PredictiveConstants.FEATURESCOLM)
        modelName = regressionInfo.get(PredictiveConstants.MODELSHEETNAME)
        labelColm = etlStats.get(PredictiveConstants.LABELCOLM)
        trainData = etlStats.get(PredictiveConstants.TRAINDATA)
        algoName = regressionInfo.get(PredictiveConstants.ALGORITHMNAME)
        locationAddress = regressionInfo.get(PredictiveConstants.LOCATIONADDRESS)
        modelId = regressionInfo.get(PredictiveConstants.MODELID)

        regParam = 0.05 if regParam == None else float(regParam)
        elasticNetPara = 1 if algoName == PredictiveConstants.LASSO_REG else 0
        ridgeLassoModelFit = \
            LinearRegression(featuresCol=featuresColm,
                             labelCol=labelColm,
                             elasticNetParam=elasticNetPara,
                             regParam=regParam,
                             predictionCol=modelName)
        regressor = ridgeLassoModelFit.fit(trainData)
        regressionStat = self.regressionEvaluation(regressor=regressor,
                                                   regressionInfo=regressionInfo,
                                                   etlStats=etlStats)

        # persisting model
        modelName = "lassoRegressionModel" if algoName == PredictiveConstants.LASSO_REG \
            else "ridgeRegressionModel"
        extention = ".parquet"
        modelStorageLocation = locationAddress + modelId.upper() + modelName.upper() + extention
        regressor.write().overwrite().save(modelStorageLocation)

        regressionStat["modelPersistLocation"] = {"modelName": modelName,
                                                  "modelStorageLocation": modelStorageLocation}

        return regressionStat

    def createGraphData(self, regressor, regressionInfo, etlStats):
        pass

    def regressionEvaluation(self, regressor, regressionInfo, etlStats):
        # right now using the linear regression method for eval
        predictiveLinearRegression = PredictiveLinearRegression()
        response = predictiveLinearRegression.regressionEvaluation(
            regressor=regressor, regressionInfo=regressionInfo, etlStats=etlStats)

        return response
