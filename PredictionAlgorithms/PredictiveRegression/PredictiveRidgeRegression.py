from pyspark.ml.regression import LinearRegression
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants
from PredictionAlgorithms.PredictiveRegression.PredictiveLinearRegression import PredictiveLinearRegression
from PredictionAlgorithms.PredictiveRegression.PredictiveRegression import PredictiveRegression

class PredictiveRidgeRegression(PredictiveRegression):

    def __init__(self):
        pass

    def ridgeRegression(self, regressionInfo):
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
        modelNameLocal = "lassoRegressionModel" if algoName == PredictiveConstants.LASSO_REG \
            else "ridgeRegressionModel"
        extention = ".parquet"
        modelStorageLocation = locationAddress + modelId.upper() + modelNameLocal.upper() + extention
        regressor.write().overwrite().save(modelStorageLocation)

        regressionStat["modelPersistLocation"] = {"modelName": modelNameLocal,
                                                  "modelStorageLocation": modelStorageLocation}

        return regressionStat

    def createGraphData(self, regressor, regressionInfo, etlStats):
        pass

    def regressionEvaluation(self, regressor, regressionInfo, etlStats):
        # using the linear regression method temporarily
        predictiveLinearRegression = PredictiveLinearRegression()
        response = predictiveLinearRegression.regressionEvaluation(
            regressor=regressor, regressionInfo=regressionInfo, etlStats=etlStats)

        return response
