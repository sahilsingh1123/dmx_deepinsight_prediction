from PredictionAlgorithms.SentimentAnalysis.TextProcessing import TextProcessing
from PredictionAlgorithms.SentimentAnalysis.SentimentAnalysis import SentimentAnalysis
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, CountVectorizer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

'''temporary'''
sparkTest = \
    SparkSession.builder.appName('DMXPredictiveAnalytics').master('local[*]').getOrCreate()
sparkTest.sparkContext.setLogLevel('ERROR')

class DecisionTreeClassifierTest(SentimentAnalysis):

    def sentimentAnalysis(self, sentimentInfoData):
        spark = sentimentInfoData.get(pc.SPARK)
        sentimentDataset = self.textPreProcessing(sentimentInfoData) # do the oneHot Encoding after that.
        textProcessing = TextProcessing(sparkSession=spark)
        sentimentDataset = textProcessing.lemmatization(sentimentDataset,pc.DMXSTOPWORDS)
        sentimentInfoData.update({pc.COLMTOENCODE: pc.DMXLEMMATIZED,
                                  pc.DATASET: sentimentDataset}) #--> colm to be used in oneHot encoding.
        sentimentInfoData = self.oneHotEncodeData(sentimentInfoData) # using the stopWords colm for now.
        sentimentInfoData = self.labelIndexing(sentimentInfoData) # after this will get the indexed label
        sentimentInfoData = self.trainModel(sentimentInfoData)

    def oneHotEncodeData(self,sentimentInfoData):
        colName = sentimentInfoData.get(pc.COLMTOENCODE)
        dataset = sentimentInfoData.get(pc.DATASET)
        vectorizedFeaturescolmName = "features" # temp fix for testing only
        dataset.drop(vectorizedFeaturescolmName)
        oneHotEncodedColName = pc.ONEHOTENCODED_ + colName
        countVectorizer = CountVectorizer(inputCol=pc.DMXSTOPWORDS,
                                          outputCol=oneHotEncodedColName).fit(dataset)
        '''oneHotEncoderPath = storageLocation + modelId.upper() + PredictiveConstants.ONEHOTENCODED.upper() + PredictiveConstants.PARQUETEXTENSION
        oneHotEncoder.write().overwrite().save(oneHotEncoderPath)
        oneHotEncoderPathMapping.update({
            PredictiveConstants.ONEHOTENCODED: oneHotEncoderPath
        })'''

        dataset = countVectorizer.transform(dataset)
        # need to store the path of count vectorizer to use at the time of performing sentiment analysis.

        '''create feature colm from encoded colm'''
        featureassembler = VectorAssembler(
            inputCols=[oneHotEncodedColName],
            outputCol=vectorizedFeaturescolmName, handleInvalid="skip")
        dataset = featureassembler.transform(dataset)
        sentimentInfoData.update({
            pc.FEATURECOLUMN: vectorizedFeaturescolmName,
            pc.DATASET: dataset
        })
        return sentimentInfoData

    def labelIndexing(self, sentimentInfoData):
        labelColm = sentimentInfoData.get(pc.LABELCOLM)
        dataset = sentimentInfoData.get(pc.DATASET)
        indexedLabel = pc.INDEXED_ + labelColm
        #check if the datatype of the col is integer or float or double. if yes then no need to do the indexing.
        '''for now converting each datatypes to string and then indexing it.'''
        dataset = dataset.withColumn(labelColm, dataset[labelColm].cast(StringType()))
        labelIndexer = StringIndexer(inputCol=labelColm, outputCol=indexedLabel,
                                     handleInvalid="keep").fit(dataset)
        dataset = labelIndexer.transform(dataset)
        #storeLabelIndexer = labelIndexer.write().overwrite().save("") # will update this later
        sentimentInfoData.update({
            pc.INDEXEDCOLM: indexedLabel,
            pc.DATASET: dataset
        })

        return sentimentInfoData

    def trainModel(self, sentimentInfoData):
        label = sentimentInfoData.get(pc.INDEXEDCOLM)
        feature = sentimentInfoData.get(pc.FEATURECOLUMN)
        dataset = sentimentInfoData.get(pc.DATASET)
        '''temp split the dataset to training and testing dataset'''
        (trainDataset, testDataset) = dataset.randomSplit([0.7,0.3])
        decisionTreeClassifier = DecisionTreeClassifier(labelCol= label, featuresCol=feature)
        decisionModel = decisionTreeClassifier.fit(trainDataset)
        # decisionModel.transform(trainDataset).groupBy("sentiment").count().show()
        predictionDataset = decisionModel.transform(testDataset)
        #calculating the accuracy of the model
        evaluator = MulticlassClassificationEvaluator(
            labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictionDataset)
        print("Test Error = %g " % (1.0 - accuracy))

        '''gbt = GBTClassifier(labelCol= label, featuresCol= feature).fit(trainDataset)
        gbtTrain = gbt.transform(trainDataset)'''



if (__name__== "__main__"):
    reviewDatasetPath = "/home/fidel/Documents/MOVIEDATASETPARQUET.parquet"
    positiveDatasetPath = "/home/fidel/Documents/POSITIVESENTIMENTDATASETPARQUET.parquet"
    negativeDatasetPath = "/home/fidel/Documents/NEGATIVESENTIMENTDATASETPARQUET.parquet"
    sentimentColName = "SentimentText"
    positiveColName = "words"
    negativeColName = "words"
    labelColName = "Sentiment"

    decisionTreeInfo = {
        pc.SENTIMENTDATASETPATH: reviewDatasetPath,
        pc.POSITIVEDATASETPATH: positiveDatasetPath,
        pc.NEGATIVEDATASETPATH: negativeDatasetPath,
        pc.SENTIMENTCOLNAME: sentimentColName,
        pc.POSITIVECOLNAME: positiveColName,
        pc.NEGATIVECOLNAME: negativeColName,
        pc.SPARK: sparkTest,
        pc.LABELCOLM : labelColName
    }
    decisionTree = DecisionTreeClassifierTest()
    decisionTree.sentimentAnalysis(decisionTreeInfo)
