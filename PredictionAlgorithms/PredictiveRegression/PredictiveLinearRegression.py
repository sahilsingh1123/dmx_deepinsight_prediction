from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants
from PredictionAlgorithms.PredictiveRegression.PredictiveRegression import PredictiveRegression
from pyspark.sql.types import *
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pUtil

# spark = \
#     SparkSession.builder.appName('predictive_Analysis').master('local[*]').getOrCreate()
# spark.sparkContext.setLogLevel('ERROR')


class PredictiveLinearRegression(PredictiveRegression):
    # global spark

    def __init__(self):
        pass

    def linearRegression(self, regressionInfo):
        # calling etl method for etl Operation
        etlStats = self.etlOperation(etlInfo=regressionInfo)

        featuresColm = etlStats.get(PredictiveConstants.FEATURESCOLM)
        modelName = regressionInfo.get(PredictiveConstants.MODELSHEETNAME)
        labelColm = etlStats.get(PredictiveConstants.LABELCOLM)
        trainData = etlStats.get(PredictiveConstants.TRAINDATA)
        locationAddress = regressionInfo.get(PredictiveConstants.LOCATIONADDRESS)
        modelId = regressionInfo.get(PredictiveConstants.MODELID)
        spark = regressionInfo.get(PredictiveConstants.SPARK)
        # global spark
        # spark = sparkSession

        linearRegressionModelfit = \
            LinearRegression(featuresCol=featuresColm, labelCol=labelColm,
                             predictionCol=modelName)
        regressor = linearRegressionModelfit.fit(trainData)
        # calling evaluation method for regression model
        regressionStat = self.regressionEvaluation(regressor=regressor,
                                                   regressionInfo=regressionInfo,
                                                   etlStats=etlStats)

        # persisting the model
        modelNameLocal = "linearRegressionModel"
        extention = ".parquet"
        modelStorageLocation = locationAddress + modelId.upper() + modelNameLocal.upper() + extention
        regressor.write().overwrite().save(modelStorageLocation)

        regressionStat["modelPersistLocation"] = {"modelName": modelNameLocal,
                                                  PredictiveConstants.MODELSTORAGELOCATION: modelStorageLocation}

        return regressionStat

    def createGraphData(self, regressor, regressionInfo, etlStats):

        # getting data from regressionInfo
        modelId = regressionInfo.get(PredictiveConstants.MODELID)
        locationAddress = regressionInfo.get(PredictiveConstants.LOCATIONADDRESS)
        modelName = regressionInfo.get(PredictiveConstants.MODELSHEETNAME)
        spark = regressionInfo.get(PredictiveConstants.SPARK)

        # getting data from etl data
        labelColm = etlStats.get(PredictiveConstants.LABELCOLM)
        testData = etlStats.get(PredictiveConstants.TESTDATA)

        trainingSummary = regressor.summary
        residualsTraining = trainingSummary.residuals

        # test and training data predicted vs actual graphdata
        trainingPredictionAllColm = trainingSummary.predictions
        trainingPredictionActual = \
            trainingPredictionAllColm.select(labelColm, modelName)
        trainingPredictionActualGraphFileName = \
            pUtil.writeToParquet(fileName="trainingPredictedVsActual",
                                 locationAddress=locationAddress,
                                 userId=modelId,
                                 data=trainingPredictionActual)
        testPredictionAllColm = regressor.transform(testData)
        testPredictionActual = \
            testPredictionAllColm.select(labelColm, modelName)
        testPredictionActualGraphFileName = \
            pUtil.writeToParquet(fileName="testPredictedVsActual",
                                 locationAddress=locationAddress,
                                 userId=modelId,
                                 data=testPredictionActual)

        # appending train and test dataset together
        # for future use only
        trainTestMerged = trainingPredictionAllColm.union(testPredictionAllColm)
        trainTestMergedFileName = \
            pUtil.writeToParquet(fileName="trainTestMerged",
                                 locationAddress=locationAddress,
                                 userId=modelId,
                                 data=trainTestMerged)
        # residual vs fitted graph
        residualsPredictiveDataTraining = \
            pUtil.residualsFittedGraph(residualsData=residualsTraining,
                                       predictionData=trainingPredictionActual,
                                       modelSheetName=modelName,
                                       spark=spark)

        residualsVsFittedGraphFileName = \
            pUtil.writeToParquet(fileName="residualsVsFitted",
                                 locationAddress=locationAddress,
                                 userId=modelId,
                                 data=residualsPredictiveDataTraining)
        # scale location plot
        sqrtStdResiduals = \
            pUtil.scaleLocationGraph(label=labelColm,
                                     predictionTargetData=trainingPredictionActual,
                                     residualsData=residualsTraining,
                                     modelSheetName=modelName,
                                     spark=spark)
        scaleLocationGraphFileName = \
            pUtil.writeToParquet(fileName="scaleLocation",
                                 locationAddress=locationAddress,
                                 userId=modelId,
                                 data=sqrtStdResiduals)
        # quantile plot
        quantileQuantileData = \
            pUtil.quantileQuantileGraph(residualsData=residualsTraining,
                                        spark=spark)

        quantileQuantileGraphFileName = \
            pUtil.writeToParquet(fileName="quantileQuantile",
                                 locationAddress=locationAddress,
                                 userId=modelId,
                                 data=quantileQuantileData)

        # creating dictionary for the graph data and summary stats
        graphNameDict = {PredictiveConstants.RESIDUALSVSFITTEDGRAPHFILENAME: residualsVsFittedGraphFileName,
                         PredictiveConstants.SCALELOCATIONGRAPHFILENAME: scaleLocationGraphFileName,
                         PredictiveConstants.QUANTILEQUANTILEGRAPHFILENAME: quantileQuantileGraphFileName,
                         PredictiveConstants.TRAININGPREDICTIONACTUALFILENAME: trainingPredictionActualGraphFileName,
                         PredictiveConstants.TESTPREDICTIONACTUALFILENAME: testPredictionActualGraphFileName}

        return graphNameDict

    def regressionEvaluation(self, regressor, regressionInfo, etlStats):
        import builtins
        round = getattr(builtins, 'round')

        modelId = regressionInfo.get(PredictiveConstants.MODELID)
        locationAddress = regressionInfo.get(PredictiveConstants.LOCATIONADDRESS)
        algoName = regressionInfo.get(PredictiveConstants.ALGORITHMNAME)
        spark = regressionInfo.get(PredictiveConstants.SPARK)

        # getting data from etl data
        labelColm = etlStats.get(PredictiveConstants.LABELCOLM)
        idNameFeaturesOrdered = etlStats.get(PredictiveConstants.IDNAMEFEATURESORDERED)

        try:
            coefficientStdErrorList = regressor.summary.coefficientStandardErrors
            coefficientStdErrorDict = {}
            statsDictName = "coefficientStdErrorDictWithName"

            coefficientStdErrorDictWithName = pUtil.statsDict(coefficientStdErrorList, coefficientStdErrorDict,
                                                              idNameFeaturesOrdered)

            pValuesList = regressor.summary.pValues
            pValuesDict = {}

            pValuesDictWithName = pUtil.statsDict(pValuesList, pValuesDict, idNameFeaturesOrdered)

            tValuesList = regressor.summary.tValues
            tValuesDict = {}

            tValuesDictWithName = pUtil.statsDict(tValuesList, tValuesDict, idNameFeaturesOrdered)

            significanceDict = {}
            for pkey, pVal in pValuesDict.items():
                if (0 <= pVal < 0.001):
                    significanceDict[pkey] = '***'
                if (0.001 <= pVal < 0.01):
                    significanceDict[pkey] = '**'
                if (0.01 <= pVal < 0.05):
                    significanceDict[pkey] = '*'
                if (0.05 <= pVal < 0.1):
                    significanceDict[pkey] = '.'
                if (0.1 <= pVal < 1):
                    significanceDict[pkey] = '-'
            significanceDictWithName = \
                pUtil.summaryTable(featuresName=idNameFeaturesOrdered,
                                   featuresStat=significanceDict)
        except:
            coefficientStdErrorDictWithName = {}
            pValuesDictWithName = {}
            tValuesDictWithName = {}
            significanceDictWithName = {}

        coefficientList = list(map(float, list(regressor.coefficients)))
        coefficientDict = {}
        coefficientDictWithName = pUtil.statsDict(coefficientList, coefficientDict, idNameFeaturesOrdered)

        # creating the table chart data
        summaryTableChartList = []
        if algoName != "lasso_reg":
            for (keyOne, valueOne), valueTwo, valueThree, valueFour, valueFive in \
                    zip(coefficientStdErrorDictWithName.items(), coefficientDictWithName.values(),
                        pValuesDictWithName.values(),
                        tValuesDictWithName.values(), significanceDictWithName.values()):
                chartList = [keyOne, valueOne, valueTwo, valueThree, valueFour, valueFive]
                summaryTableChartList.append(chartList)
            schemaSummaryTable = StructType([StructField("Column_Name", StringType(), True),
                                             StructField("std_Error", DoubleType(), True),
                                             StructField("coefficient", DoubleType(), True),
                                             StructField("P_value", DoubleType(), True),
                                             StructField("T_value", DoubleType(), True),
                                             StructField("significance", StringType(), True)])

        if (coefficientStdErrorDictWithName == {} or algoName == "lasso_reg"):
            for (keyOne, valueOne) in coefficientDictWithName.items():
                chartList = [keyOne, valueOne]
                summaryTableChartList.append(chartList)

            schemaSummaryTable = StructType([StructField("Column_Name", StringType(), True),
                                             StructField("coefficient", DoubleType(), True)])

        summaryTableChartData = spark.createDataFrame(summaryTableChartList, schema=schemaSummaryTable)  # fix this
        summaryTableChartDataFileName = \
            pUtil.writeToParquet(fileName="summaryTableChart",
                                 locationAddress=locationAddress,
                                 userId=modelId,
                                 data=summaryTableChartData)

        # creating the equation for the regression model
        intercept = round(regressor.intercept, 4)
        equation = labelColm, "=", intercept, "+"
        for feature, coeff in zip(idNameFeaturesOrdered.values(), coefficientDict.values()):
            coeffFeature = coeff, "*", feature, "+"
            equation += coeffFeature
        equation = list(equation[:-1])

        # training summary
        trainingSummary = regressor.summary
        RMSE = round(trainingSummary.rootMeanSquaredError, 4)
        MAE = round(trainingSummary.meanAbsoluteError, 4)
        MSE = round(trainingSummary.meanSquaredError, 4)
        rSquare = round(trainingSummary.r2, 4)
        adjustedRSquare = round(trainingSummary.r2adj, 4)
        degreeOfFreedom = trainingSummary.degreesOfFreedom
        explainedVariance = round(trainingSummary.explainedVariance, 4)
        totalNumberOfFeatures = regressor.numFeatures
        residualsTraining = trainingSummary.residuals  # sparkDataframe

        summaryStats = {PredictiveConstants.RMSE: RMSE, PredictiveConstants.MSE: MSE,
                        PredictiveConstants.MAE: MAE, PredictiveConstants.RSQUARE: rSquare,
                        PredictiveConstants.ADJRSQUARE: adjustedRSquare,
                        PredictiveConstants.INTERCEPT: intercept,
                        PredictiveConstants.DOF: degreeOfFreedom,
                        PredictiveConstants.EXPLAINEDVARIANCE: explainedVariance,
                        PredictiveConstants.TOTALFEATURES: totalNumberOfFeatures}

        summaryTable = {"summaryTableChartDataFileName": summaryTableChartDataFileName}

        # call the create graph data method
        graphDataInfo = self.createGraphData(regressor=regressor, regressionInfo=regressionInfo, etlStats=etlStats)

        response = {PredictiveConstants.GRAPHDATA: graphDataInfo,
                    PredictiveConstants.STATDATA: summaryStats,
                    PredictiveConstants.TABLEDATA: summaryTable,
                    PredictiveConstants.EQUATION: equation}

        return response
