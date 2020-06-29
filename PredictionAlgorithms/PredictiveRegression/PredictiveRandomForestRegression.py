from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants
from PredictionAlgorithms.PredictiveRegression.PredictiveRegression import PredictiveRegression
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pUtil
from pyspark.ml.evaluation import RegressionEvaluator


class PredictiveRandomForestRegression(PredictiveRegression):

    def __init__(self):
        pass

    def randomForestRegression(self, regressionInfo):
        etlStats = self.etlOperation(etlInfo=regressionInfo)

        featuresColm = etlStats.get(PredictiveConstants.FEATURESCOLM)
        modelName = regressionInfo.get(PredictiveConstants.MODELSHEETNAME)
        labelColm = etlStats.get(PredictiveConstants.LABELCOLM)
        trainData = etlStats.get(PredictiveConstants.TRAINDATA)
        locationAddress = regressionInfo.get(PredictiveConstants.LOCATIONADDRESS)
        modelId = regressionInfo.get(PredictiveConstants.MODELID)

        randomForestRegressorModelFit = \
            RandomForestRegressor(labelCol=labelColm,
                                  featuresCol=featuresColm,
                                  numTrees=10, predictionCol=modelName)
        regressor = randomForestRegressorModelFit.fit(trainData)
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
        # getting data from regressionInfo
        modelId = regressionInfo.get(PredictiveConstants.MODELID)
        locationAddress = regressionInfo.get(PredictiveConstants.LOCATIONADDRESS)
        modelName = regressionInfo.get(PredictiveConstants.MODELSHEETNAME)
        spark = regressionInfo.get(PredictiveConstants.SPARK)

        # getting data from etl data
        labelColm = etlStats.get(PredictiveConstants.LABELCOLM)
        trainData = etlStats.get(PredictiveConstants.TRAINDATA)
        testData = etlStats.get(PredictiveConstants.TESTDATA)

        trainPredictedData = regressor.transform(trainData)
        testPredictedData = regressor.transform(testData)
        # training Actual vs Predicted dataset
        trainingPredictionActual = \
            trainPredictedData.select(labelColm, modelName)
        trainingPredictionActualGraphFileName = \
            pUtil.writeToParquet(fileName="trainingPredictedVsActualEnsemble",
                                 locationAddress=locationAddress,
                                 userId=modelId,
                                 data=trainingPredictionActual)
        # test Actual Vs Predicted dataset
        testPredictionActual = \
            testPredictedData.select(labelColm, modelName)
        testPredictionActualGraphFileName = \
            pUtil.writeToParquet(fileName="testPredictedVsActualEnsemble",
                                 locationAddress=locationAddress,
                                 userId=modelId,
                                 data=testPredictionActual)
        # creating the residual vs fitted graph data
        residualDataColm = trainingPredictionActual.withColumn('residuals',
                                                               col(labelColm) - col(modelName))
        residualDataColm = residualDataColm.select('residuals')
        residualsPredictiveDataTraining = \
            pUtil.residualsFittedGraph(residualsData=residualDataColm,
                                       predictionData=trainingPredictionActual,
                                       modelSheetName=modelName,
                                       spark=spark)
        residualsVsFittedGraphFileName = \
            pUtil.writeToParquet(fileName="residualsVsFittedEnsemble",
                                 locationAddress=locationAddress,
                                 userId=modelId,
                                 data=residualsPredictiveDataTraining)

        graphNameDict = {PredictiveConstants.RESIDUALSVSFITTEDGRAPHFILENAME: residualsVsFittedGraphFileName,
                         PredictiveConstants.TRAININGPREDICTIONACTUALFILENAME: trainingPredictionActualGraphFileName,
                         PredictiveConstants.TESTPREDICTIONACTUALFILENAME: testPredictionActualGraphFileName}
        return graphNameDict

    def regressionEvaluation(self, regressor, regressionInfo, etlStats):
        modelName = regressionInfo.get(PredictiveConstants.MODELSHEETNAME)

        # getting data from etl data
        labelColm = etlStats.get(PredictiveConstants.LABELCOLM)
        trainData = etlStats.get(PredictiveConstants.TRAINDATA)
        idNameFeaturesOrdered = etlStats.get(PredictiveConstants.IDNAMEFEATURESORDERED)

        trainPredictedData = regressor.transform(trainData)
        metricsList = ['r2', 'rmse', 'mse', 'mae']
        trainDataMetrics = {}
        metricName = ''
        for metric in metricsList:
            if metric.__eq__("r2"):
                metricName = PredictiveConstants.RSQUARE
            elif metric.__eq__("rmse"):
                metricName = PredictiveConstants.RMSE
            elif metric.__eq__("mse"):
                metricName = PredictiveConstants.MSE
            elif metric.__eq__("mae"):
                metricName = PredictiveConstants.MAE
            evaluator = RegressionEvaluator(labelCol=labelColm,
                                            predictionCol=modelName,
                                            metricName=metric)
            metricValue = evaluator.evaluate(trainPredictedData)
            trainDataMetrics[metricName] = metricValue

        # summary stats
        noTrees = regressor.getNumTrees
        treeWeights = regressor.treeWeights
        treeNodes = list(regressor.trees)
        totalNoNodes = regressor.totalNumNodes
        debugString = regressor.toDebugString

        debugString = str(debugString).splitlines()

        featuresImportance = list(regressor.featureImportances)
        featuresImportance = [round(x, 4) for x in featuresImportance]
        featuresImportanceDict = {}
        for importance in featuresImportance:
            featuresImportanceDict[featuresImportance.index(importance)] = importance

        featuresImportanceDictWithName = \
            pUtil.summaryTable(featuresName=idNameFeaturesOrdered,
                               featuresStat=featuresImportanceDict)

        trainDataMetrics["No Trees"] = noTrees
        trainDataMetrics["Total Nodes"] = totalNoNodes

        summaryStats = {'noTrees': noTrees,
                        'treeWeights': treeWeights,
                        'totalNodes': totalNoNodes,
                        'featuresImportance': featuresImportanceDictWithName,
                        'metrics': trainDataMetrics,
                        'debugString': debugString,
                        }

        graphDataInfo = self.createGraphData(regressor, regressionInfo, etlStats)

        response = {PredictiveConstants.STATDATA: summaryStats,
                    PredictiveConstants.GRAPHDATA: graphDataInfo}

        return response
