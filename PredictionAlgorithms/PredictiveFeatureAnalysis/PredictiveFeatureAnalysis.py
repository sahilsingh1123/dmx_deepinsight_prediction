from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor

from PredictionAlgorithms.PredictiveConstants import PredictiveConstants
from PredictionAlgorithms.PredictiveDataTransformation import PredictiveDataTransformation
from PredictionAlgorithms.PredictiveStatisticalTest import PredictiveStatisticalTest
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities


class PredictiveFeatureAnalysis():
    def __init__(self):
        pass

    def featureSelection(self, predictiveData):
        algoName = predictiveData.get(PredictiveConstants.ALGORITHMNAME)

        etlStats = PredictiveUtilities.performETL(predictiveData)
        featureStatistics = self.featureStats(etlStats, predictiveData)

        categoryColmStats = etlStats.get(PredictiveConstants.CATEGORYCOLMSTATS)
        idNameFeaturesOrdered = etlStats.get(PredictiveConstants.IDNAMEFEATURESORDERED)

        featureAnalysis = self.featureAnalysis(etlStats, algoName)
        randomForestModelFit = featureAnalysis.get(PredictiveConstants.RANDOMFORESTMODEL)

        featureImportanceData = self.getFeatureImportance(randomForestModelFit, idNameFeaturesOrdered)

        featureImportanceDict = featureImportanceData.get(PredictiveConstants.FEATURESIMPORTANCEDICT)
        featureImportance = featureImportanceData.get(PredictiveConstants.FEATURE_IMPORTANCE)
        summaryDict = featureStatistics.get(PredictiveConstants.SUMMARYDICT)
        featuresStatsDict = featureStatistics.get(PredictiveConstants.FEATURESSTATSDICT)
        keyStatsTest = featureAnalysis.get(PredictiveConstants.KEYSTATSTEST)
        statisticalTestResult = featureAnalysis.get(PredictiveConstants.STATISTICALTESTRESULT)

        responseData = {
            PredictiveConstants.FEATURE_IMPORTANCE: featureImportance,
            keyStatsTest: statisticalTestResult,
            PredictiveConstants.SUMMARYDICT: summaryDict,
            PredictiveConstants.CATEGORICALSUMMARY: categoryColmStats,
            PredictiveConstants.FEATURESIMPORTANCEDICT: featureImportanceDict,
            PredictiveConstants.FEATURESSTATSDICT: featuresStatsDict
        }

        return responseData

    def featureAnalysis(self, etlStats, algoName):

        numericalFeatures = etlStats.get(PredictiveConstants.NUMERICALFEATURES)
        label = etlStats.get(PredictiveConstants.LABELCOLM)
        dataset = etlStats.get(PredictiveConstants.DATASET)
        featuresColm = etlStats.get(PredictiveConstants.FEATURESCOLM)
        indexedFeatures = etlStats.get(PredictiveConstants.INDEXEDFEATURES)
        maxCategories = etlStats.get(PredictiveConstants.MAXCATEGORIES)
        categoricalFeatures = etlStats.get(PredictiveConstants.CATEGORICALFEATURES)

        trainData, testData = dataset.randomSplit([0.80, 0.20], seed=40)

        keyStatsTest = ''
        statisticalTestResult = {}
        randomForestModel = object
        if algoName == PredictiveConstants.RANDOMREGRESSOR:
            statisticalTestObj = PredictiveStatisticalTest(dataset=dataset,
                                                           features=numericalFeatures,
                                                           labelColm=label)
            statisticalTestResult = statisticalTestObj.pearsonTest()
            randomForestModel = \
                RandomForestRegressor(labelCol=label,
                                      featuresCol=featuresColm,
                                      numTrees=10)
            keyStatsTest = "pearson_test_data"
        if algoName == PredictiveConstants.RANDOMCLASSIFIER:
            statisticalTestObj = PredictiveStatisticalTest(dataset=dataset,
                                                           features=indexedFeatures,
                                                           labelColm=label)
            statisticalTestResult = \
                statisticalTestObj.chiSquareTest(categoricalFeatures=categoricalFeatures,
                                                 maxCategories=maxCategories)
            randomForestModel = RandomForestClassifier(labelCol=label,
                                                       featuresCol=featuresColm,
                                                       numTrees=10)
            keyStatsTest = "ChiSquareTestData"
        randomForestModelFit = randomForestModel.fit(trainData)

        featureAnalysis = {
            PredictiveConstants.RANDOMFORESTMODEL: randomForestModelFit,
            PredictiveConstants.KEYSTATSTEST: keyStatsTest,
            PredictiveConstants.STATISTICALTESTRESULT: statisticalTestResult
        }
        return featureAnalysis

    def featureStats(self, etlStats, predictiveData):
        numericalFeatures = etlStats.get(PredictiveConstants.NUMERICALFEATURES)
        label = etlStats.get(PredictiveConstants.LABELCOLM)
        dataset = etlStats.get(PredictiveConstants.DATASET)
        categoricalFeatures = etlStats.get(PredictiveConstants.CATEGORICALFEATURES)
        categoryColmStats = etlStats.get(PredictiveConstants.CATEGORYCOLMSTATS)

        locationAddress = predictiveData.get(PredictiveConstants.LOCATIONADDRESS)
        featureId = predictiveData.get(PredictiveConstants.MODELID)

        # statistics
        columnListForfeaturesStats = numericalFeatures.copy()
        columnListForfeaturesStats.insert(0, label)
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataStatsResult = \
            dataTransformationObj.dataStatistics(categoricalFeatures=categoricalFeatures,
                                                 numericalFeatures=columnListForfeaturesStats,
                                                 categoricalColmStat=categoryColmStats)
        summaryDict = dataStatsResult

        # creating the dataset for statschart visualization in features selection chart
        datasetForStatsChart = dataset.select(columnListForfeaturesStats)
        datasetForStatsChartFileName = \
            PredictiveUtilities.writeToParquet(fileName="datasetForStatsChart",
                                               locationAddress=locationAddress,
                                               userId=featureId,
                                               data=datasetForStatsChart)

        featuresStatsDict = {"columnsName": columnListForfeaturesStats,
                             "datasetFileName": datasetForStatsChartFileName}

        featureStatistics = {
            PredictiveConstants.SUMMARYDICT: summaryDict,
            PredictiveConstants.FEATURESSTATSDICT: featuresStatsDict
        }

        return featureStatistics

    def getFeatureImportance(self, randomForestModelFit, idNameFeaturesOrdered):
        import builtins
        round = getattr(builtins, 'round')

        featuresImportance = list(randomForestModelFit.featureImportances)
        featuresImportance = [round(x, 4) for x in featuresImportance]
        featuresImportanceDict = {}
        for importance in featuresImportance:
            featuresImportanceDict[featuresImportance.index(importance)] = round(importance, 4)

        featuresImportanceDictWithName = \
            PredictiveUtilities.summaryTable(featuresName=idNameFeaturesOrdered,
                                             featuresStat=featuresImportanceDict)

        featuresColmList = idNameFeaturesOrdered
        feat = []
        for val in featuresColmList.values():
            feat.append(val)
        feature_imp = {
            PredictiveConstants.FEATURE_IMPORTANCE: featuresImportance,
            "feature_column": feat
        }

        featuresImportanceData = {
            PredictiveConstants.FEATURE_IMPORTANCE: feature_imp,
            PredictiveConstants.FEATURESIMPORTANCEDICT: featuresImportanceDictWithName
        }
        return featuresImportanceData
