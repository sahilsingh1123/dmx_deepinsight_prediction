from PredictionAlgorithms.SentimentAnalysis.TextProcessing import TextProcessing
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu
from sparknlp.base import PipelineModel
from sparknlp.pretrained import ViveknSentimentModel
from sparknlp.annotator import StopWordsCleaner
from pyspark.ml import Pipeline
from pyspark.sql.functions import array_join
import json
import sparknlp
from pyspark.sql import SparkSession

# sparkTest = SparkSession.builder.appName("sentimentPrediction").master("local[*]").getOrCreate()
# sparkTest.sparkContext.setLogLevel('ERROR')

sparkTest = sparknlp.start()

class SANLPPrediction(object):
    def __init__(self):
        pass

    def prediction(self, infoData):
        infoData = self.textProcessing(infoData)

    def textProcessing(self, infoData):
        sentimentCol = infoData.get(pc.SENTIMENTCOLNAME)
        dataset = self.cleanData(infoData)
        dataset = dataset.withColumnRenamed(sentimentCol, "text")
        documentPipeline = infoData.get(pc.DOCUMENTPRETRAINEDPIPELINE)
        loadedDocPipeline = PipelineModel.load(documentPipeline)
        # dataset = loadedDocPipeline.transform(dataset)

        # removing stopWords
        stopWordsRemover = StopWordsCleaner().setInputCols(["lemma"]) \
            .setOutputCol(pc.DMXSTOPWORDS)
        # dataset = stopWordsRemover.transform(dataset)

        textProcessingPipeline = Pipeline(stages=[loadedDocPipeline, stopWordsRemover])
        dataset = textProcessingPipeline.fit(dataset).transform(dataset)

        infoData.update({pc.DATASET:dataset})

        infoData = self.performSentiment(infoData)

        return infoData

    def performSentiment(self, infoData):
        dataset = infoData.get(pc.DATASET)
        sentimentModelPath = (infoData.get(pc.SPARKNLPPATHMAPPING)).get(pc.SENTIMENTMODEL)
        viveknSentimentModel = ViveknSentimentModel.load(sentimentModelPath)
        dataset = viveknSentimentModel.transform(dataset)
        dataset = dataset.withColumn("viveknSentiment", array_join("viveknSentiment.result", ""))
        infoData.update({pc.DATASET: dataset})

        return infoData




    def cleanData(self, infoData):
        datasetPath = infoData.get(pc.SENTIMENTDATASETPATH)
        sentimentCol = infoData.get(pc.SENTIMENTCOLNAME)
        spark = infoData.get(pc.SPARK)
        dataset = spark.read.parquet(datasetPath)
        textProcessing = TextProcessing(sparkSession=spark)
        dataset = textProcessing.replaceSpecialChar(dataset, sentimentCol)
        return dataset

if (__name__ == "__main__"):
    infoData = json.load(open("/home/fidel/Documents/sparkNLP.json"))
    datasetPath = "/home/fidel/Documents/KNIMETESTDATASET.parquet"
    infoData.update({pc.SPARK: sparkTest,
                     pc.SENTIMENTDATASETPATH: datasetPath})
    sentimentPrediction = SANLPPrediction()
    sentimentPrediction.prediction(infoData)
