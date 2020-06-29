from PredictionAlgorithms.SentimentAnalysis.SAMachineLearning import SAMachineLearning
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark
import json

# sparkTest = \
#     SparkSession.builder.appName('DMXPredictiveAnalytics').master('local[*]').getOrCreate()
# sparkTest.sparkContext.setLogLevel('ERROR')

conf = pyspark.SparkConf().setAppName("predictive_Analysis").setMaster("spark://fidel-Latitude-E5570:7077") #need to get the master url from the java end.
sc = SparkContext(conf=conf)
sparkTest = SparkSession(sc)


class SAGradientBoostClassifier(SAMachineLearning):
    def sentimentData(self, sentimentDataInfo):
        sentimentDataInfo = self.sentimentAnalysis(sentimentDataInfo)
        sentimentDataInfo = self.trainModel(sentimentDataInfo)
        sentimentDataInfo = self.invertIndexColm(sentimentDataInfo)
        modelName = sentimentDataInfo.get(pc.MODELSHEETNAME)
        storagePath = sentimentDataInfo.get(pc.STORAGELOCATION)
        # store the data in json format
        sentimentDataInfo.pop(pc.SPARK, "None")
        sentimentDataInfo.pop(pc.DATASET, "None")
        sentimentDataInfo.pop(pc.TESTDATA, "None")
        sentimentDataInfo.pop(pc.TRAINDATA, "None")
        sentimentDataInfo.pop(pc.MODEL, "None")
        json.dump(sentimentDataInfo, open(storagePath + modelName + ".json", 'w'))

    def trainModel(self, infoData):
        label = infoData.get(pc.INDEXEDCOLM)
        feature = infoData.get(pc.FEATURESCOLM)
        dataset = infoData.get(pc.DATASET)
        predictionColm = infoData.get(pc.PREDICTIONCOLM)

        '''temp split the dataset to training and testing dataset'''
        (trainDataset, testDataset) = dataset.randomSplit([0.80, 0.20], seed=0)
        gradientBoostClassifier = GBTClassifier(labelCol= label, featuresCol= feature,
                                                predictionCol=predictionColm).fit(trainDataset)
        trainDataset = gradientBoostClassifier.transform(trainDataset)
        testDataset = gradientBoostClassifier.transform(testDataset)
        infoData.update({
            pc.TESTDATA: testDataset,
            pc.TRAINDATA: trainDataset,
            pc.MODEL: gradientBoostClassifier
        })

        infoData = self.storeModel(infoData)
        infoData = self.evaluation(infoData)

        return infoData

    def evaluation(self, infoData):
        labelCol = infoData.get(pc.INDEXEDCOLM)
        predictionColm = infoData.get(pc.PREDICTIONCOLM)
        dataset = infoData.get(pc.TESTDATA) # for now evaluating for test data only.
        # calculating the accuracy of the model
        evaluator = MulticlassClassificationEvaluator(
            labelCol=labelCol, predictionCol=predictionColm, metricName="accuracy")
        accuracy = evaluator.evaluate(dataset)
        print("Test Error = %g " % (1.0 - accuracy))

        return infoData

    def storeModel(self, infoData):
        storagePath = infoData.get(pc.STORAGELOCATION)
        modelName = infoData.get(pc.MODELSHEETNAME)
        model = infoData.get(pc.MODEL)
        modelPath = storagePath + modelName
        model.write().overwrite().save(modelPath)
        infoData.update({pc.MODELSTORAGELOCATION: modelPath})

        return infoData

if (__name__ == "__main__"):
    algoName = "GradientBoostClassifier"
    isNgram = False
    ngramPara = 2
    # reviewDatasetPath = "/home/fidel/Documents/IMDBSAMPLE.parquet"
    reviewDatasetPath = "/home/fidel/Documents/KNIMETRAININGDATASET.parquet"
    # reviewDatasetPath = "/home/fidel/Documents/KNIMETESTDATASET.parquet"

    sentimentColName = "Text"
    sentimentModelName = "movieReviewGradientBoost"
    storageLocation = "/home/fidel/Documents/"
    # reviewDatasetPath = "/home/fidel/Documents/MOVIEDATASETPARQUET.parquet"
    positiveDatasetPath = "/home/fidel/Documents/POSITIVESENTIMENTDATASETPARQUET.parquet"
    negativeDatasetPath = "/home/fidel/Documents/NEGATIVESENTIMENTDATASETPARQUET.parquet"
    # sentimentColName = "SentimentText"
    positiveColName = "words"
    negativeColName = "words"
    labelColName = "Sentiment"
    predictionColm = pc.PREDICTION_ + sentimentModelName
    indexerPathMapping = {}
    encoderPathMapping = {}

    decisionTreeInfo = {
        pc.SENTIMENTDATASETPATH: reviewDatasetPath,
        pc.POSITIVEDATASETPATH: positiveDatasetPath,
        pc.NEGATIVEDATASETPATH: negativeDatasetPath,
        pc.SENTIMENTCOLNAME: sentimentColName,
        pc.POSITIVECOLNAME: positiveColName,
        pc.NEGATIVECOLNAME: negativeColName,
        pc.SPARK: sparkTest,
        pc.LABELCOLM: labelColName,
        pc.STORAGELOCATION: storageLocation,
        pc.INDEXERPATHMAPPING: indexerPathMapping,
        pc.PREDICTIONCOLM: predictionColm,
        pc.MODELSHEETNAME: sentimentModelName,
        pc.ISNGRAM: isNgram,
        pc.NGRAMPARA: ngramPara,
        pc.ONEHOTENCODERPATHMAPPING: encoderPathMapping,
        pc.ALGORITHMNAME: algoName
    }
    gradientBoost = SAGradientBoostClassifier()
    gradientBoost.sentimentData(decisionTreeInfo)
