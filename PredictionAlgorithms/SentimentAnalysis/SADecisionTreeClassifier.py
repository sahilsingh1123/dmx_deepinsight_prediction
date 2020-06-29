from PredictionAlgorithms.SentimentAnalysis.SAMachineLearning import SAMachineLearning
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql import SparkSession
import sparknlp

# sparkTest = \
#     SparkSession.builder.appName('DMXPredictiveAnalytics').master('local[*]').getOrCreate()
# sparkTest.sparkContext.setLogLevel('ERROR')
sparkTest = sparknlp.start()


class SADecisionTreeClassifier(SAMachineLearning):

    def sentimentData(self, sentimentDataInfo):

        sentimentDataInfo = self.sentimentAnalysis(sentimentDataInfo)
        sentimentDataInfo = self.trainModel(sentimentDataInfo)
        sentimentDataInfo = self.invertIndexColm(sentimentDataInfo)
        modelName = sentimentDataInfo.get(pc.MODELSHEETNAME)
        storagePath = sentimentDataInfo.get(pc.STORAGELOCATION)
        jsonStorageLocation = storagePath + modelName
        #--sahil store the data in json format --> write the separate method for this.
        sentimentDataInfo.pop(pc.SPARK, "None")
        sentimentDataInfo.pop(pc.DATASET, "None")
        sentimentDataInfo.pop(pc.TESTDATA, "None")
        sentimentDataInfo.pop(pc.TRAINDATA, "None")
        sentimentDataInfo.pop(pc.MODEL, "None")
        # json.dump(sentimentDataInfo, open(storagePath + modelName + ".json", 'w'))
        pu.writeToJson(jsonStorageLocation, sentimentDataInfo)


    def trainModel(self, infoData):
        label = infoData.get(pc.INDEXEDCOLM)
        feature = infoData.get(pc.FEATURESCOLM)
        dataset = infoData.get(pc.DATASET)
        predictionColm = infoData.get(pc.PREDICTIONCOLM)

        '''temp split the dataset to training and testing dataset'''
        (trainDataset, testDataset) = dataset.randomSplit([0.80, 0.20], seed=0)
        decisionTreeClassifier = DecisionTreeClassifier(labelCol=label, featuresCol=feature,
                                                        predictionCol=predictionColm)
        decisionModel = decisionTreeClassifier.fit(trainDataset)
        trainDataset = decisionModel.transform(trainDataset)
        testDataset = decisionModel.transform(testDataset)
        infoData.update({
            pc.TESTDATA: testDataset,
            pc.TRAINDATA: trainDataset,
            pc.MODEL: decisionModel
        })
        infoData = self.storeModel(infoData)
        infoData = self.evaluation(infoData)

        return infoData

    def evaluation(self, infoData):
        labelCol = infoData.get(pc.INDEXEDCOLM)
        predictionColm = infoData.get(pc.PREDICTIONCOLM)
        dataset = infoData.get(pc.TESTDATA)
        evaluator = MulticlassClassificationEvaluator(
            labelCol=labelCol, predictionCol=predictionColm, metricName="accuracy")
        accuracy = evaluator.evaluate(dataset)
        print("Test Error = %g " % (1.0 - accuracy)) # sahil- for temp only

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
    sentimentModelName = "movieReviewDecisionTree"
    storageLocation = "/home/fidel/Documents/"
    isNgram = False
    ngramPara = 2
    # reviewDatasetPath = "/home/fidel/Documents/MOVIEDATASETPARQUET.parquet"
    # reviewDatasetPath = "/home/fidel/Documents/IMDBSAMPLE.parquet"
    # reviewDatasetPath = "/home/fidel/Documents/KNIMETRAININGDATASET.parquet"
    reviewDatasetPath = "/home/fidel/Documents/KNIMETESTDATASET.parquet"

    positiveDatasetPath = "/home/fidel/Documents/POSITIVESENTIMENTDATASETPARQUET.parquet"
    negativeDatasetPath = "/home/fidel/Documents/NEGATIVESENTIMENTDATASETPARQUET.parquet"
    # sentimentColName = "SentimentText"
    sentimentColName = "Text"
    positiveColName = "words"
    negativeColName = "words"
    labelColName = "Sentiment"
    predictionColm = pc.PREDICTION_ + sentimentModelName
    indexerPathMapping = {}
    encoderPathMapping = {}
    algoName = "DecisionTreeClassifier"

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
    decisionTree = SADecisionTreeClassifier()
    decisionTree.sentimentData(decisionTreeInfo)
