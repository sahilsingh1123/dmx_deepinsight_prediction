from sparknlp.pretrained import PretrainedPipeline
from sparknlp.base import PipelineModel
from sparknlp.annotator import StopWordsCleaner
from sparknlp.pretrained import ViveknSentimentModel, ViveknSentimentApproach
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when, array_join
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.SentimentAnalysis.TextProcessing import TextProcessing
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu
import sparknlp

"""get the latest jars from the maven 
https://mvnrepository.com/artifact/com.johnsnowlabs.nlp/spark-nlp_2.11/2.4.3
"""

#'JohnSnowLabs:spark-nlp:2.4.3'
#'com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.3'
#    .config('spark.jars.packages', ','.join(packages)) \
# packages = [
#     'com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.3'
# ]

# full jar - sparknlpFATjar.jar

# from pyspark.sql import SparkSession
#
# sparkTest = SparkSession \
#     .builder \
#     .appName("spark_nlp_sentimentAnalysis") \
#     .config("spark.jars", "/home/fidel/cache_pretrained/sparknlpFATjar.jar")\
#     .getOrCreate()
# sparkTest = SparkSession.builder.appName("TEst").getOrCreate()

# sparkTest = sparknlp.start()


class SparkNLP():
    def __init__(self):
        pass

    def sentimentAnalysis(self, infoData):
        infoData = self.textProcessing(infoData)
        infoData = self.viveknSentimentAnalysis(infoData)
        storagePath = infoData.get(pc.STORAGELOCATION)
        modelName = infoData.get(pc.MODELSHEETNAME)
        modelPath = storagePath + modelName
        #writing the info data in the json file for using in sentiment prediction.
        """drop the obj which is not jsonSerializable"""
        infoData.pop(pc.SPARK, "None")
        infoData.pop(pc.DATASET, "None")
        infoData.pop(pc.TESTDATA, "None")
        pu.writeToJson(modelPath, infoData)
        print("success")

    @staticmethod
    def textProcessing(infoData):
        # read the dataset and remove all the special characters and update the dataset.
        dataset = SparkNLP.cleanData(infoData)
        sentimentCol = infoData.get(pc.SENTIMENTCOLNAME)
        storagePath = infoData.get(pc.STORAGELOCATION)
        modelName = infoData.get(pc.ALGORITHMNAME)
        documentPipeline = infoData.get(pc.DOCUMENTPRETRAINEDPIPELINE)
        dataset = dataset.withColumnRenamed(sentimentCol, "text")
        # explainDocument = PretrainedPipeline("explain_document_dl", "en") #only need to download it once.
        loadedDocumentPipeline = PipelineModel.load(documentPipeline)
        dataset = loadedDocumentPipeline.transform(dataset)

        #loading the document model for use at the time of prediction of sentiment analysis.
        # documentPipelinePath = storagePath + modelName
        # loadedDocumentPipeline.write().overwrite().save(documentPipelinePath)
        # infoData.get(infoData.get(pc.SPARKNLPPATHMAPPING).update({pc.DOCUMENTMODEL: documentPipelinePath}))

        #remove this at final stage.
        stopWordsRemover = StopWordsCleaner().setInputCols(["lemma"])\
            .setOutputCol(pc.DMXSTOPWORDS)
        dataset = stopWordsRemover.transform(dataset)

        """need to implement the pipeline on later stage"""
        # pipeline = Pipeline(stages=[explainDocument, stopWordsRemover])
        # dataset = pipeline.transform(dataset)
        # dataset = self.changeSentimentVal(dataset, labelCol)

        infoData.update({pc.DATASET: dataset})
        return infoData

    def stringToIndex(self, infoData):

        dataset = infoData.get(pc.TESTDATA)
        stringIndexerOriginal = StringIndexer(inputCol="original_sentiment", outputCol="original_sentiment_indexed")
        stringIndexerPredicted = StringIndexer(inputCol="viveknSentiment", outputCol="viveknSentiment_indexed")

        indexerPipeline = Pipeline(stages=[stringIndexerOriginal, stringIndexerPredicted])
        indexerPipelineFit = indexerPipeline.fit(dataset)
        dataset = indexerPipelineFit.transform(dataset)

        # if the indexing of both the column doesnot matches then match the index value of these.
        dataset = dataset.withColumn("viveknSentiment_indexed",
                                                       when(dataset["viveknSentiment_indexed"] == 0.0, 1.0)
                                                       .when(dataset["viveknSentiment_indexed"] == 1.0, 0.0))
        infoData.update({pc.TESTDATA: dataset,
                         pc.INDEXEDCOLM: "original_sentiment_indexed",
                         pc.PREDICTIONCOLM:"viveknSentiment_indexed"})
        return infoData
    @staticmethod
    def cleanData(infoData):
        spark = infoData.get(pc.SPARK)
        sentimentCol = infoData.get(pc.SENTIMENTCOLNAME)
        dataset = infoData.get(pc.DATASET)
        textProcessing = TextProcessing(sparkSession=spark)
        dataset = textProcessing.replaceSpecialChar(dataset, sentimentCol)
        return dataset

    def changeSentimentVal(self, dataset, labelCol):
        dataset = dataset.withColumn("original_sentiment",
                                     when(dataset[labelCol] == "POS", "Positive")
                                     .when(dataset[labelCol] == "NEG", "Pegative"))
        return dataset

    def viveknSentimentAnalysis(self, infoData):
        dataset = infoData.get(pc.DATASET)
        labelCol = infoData.get(pc.LABELCOLM)
        dataset = self.changeSentimentVal(dataset, labelCol)
        (trainDataset, testDataset) = dataset.randomSplit([0.80, 0.20], seed=0)
        viveknSentiment = ViveknSentimentApproach().setInputCols(["document", pc.DMXSTOPWORDS])\
            .setOutputCol("viveknSentiment").setSentimentCol("original_sentiment")
        viveknSentimentModel = viveknSentiment.fit(trainDataset)
        testDatasetPrediction = viveknSentimentModel.transform(testDataset)

        #storing the model at a location for future use in case of prediction of sentiment analysis.
        """you will get the list of all trained models and pretrained pipelines for using in the prediction of sentiment"""
        storagePath = infoData.get(pc.STORAGELOCATION)
        modelName = "testViveknSentiment"  #sahil - temporary only
        modelPath = storagePath + modelName
        viveknSentimentModel.write().overwrite().save(modelPath)
        infoData.get(infoData.get(pc.SPARKNLPPATHMAPPING).update({pc.SENTIMENTMODEL: modelPath}))


        #convert back the column type to the string format
        testDatasetPrediction = testDatasetPrediction.withColumn("viveknSentiment", array_join("viveknSentiment.result", ""))
        infoData.update({pc.TESTDATA: testDatasetPrediction})

        # need to coverts both the colms original sentiment and predicted sentiment for evaluation.
        infoData = self.evaluation(self.stringToIndex(infoData))

        """
        --> first check if the indexing is matching with label/original sentiment if not then match with the below method.
        finalDatasetTest = finalDatasetTest.withColumn("finalDataset_indexed",
                 when(finalDatasetTest["finalDataset_indexed"] == 0.0, 1.0)
                 .when(finalDatasetTest["finalDataset_indexed"] == 1.0, 0.0))
        """
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
