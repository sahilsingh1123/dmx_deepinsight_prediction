from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import IDF, CountVectorizer, CountVectorizerModel
from pyspark.ml.clustering import LDA
from PredictionAlgorithms.SentimentAnalysis.TextProcessing import TextProcessing
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

"""
this class will contains the code related to
text analytics from scratch to the first code
which has been discussed with vikram sir in a
meeting.
"""

# haeir dataset from hadoop - SAHIL_HAEIRENGLISHONLY.parquet

''' approach---
-- first read the parquet
-- then take it to text processing class where we do all
-- cleaning and stop words removal part ... also the lemmatization part.
-- get the lemmatized data from the text processing class-
-- which will then work as cleaned text for using it in 
-- TF-IDF part where u will find the weightage of every words in the doc.
 '''

sparkTest = \
    SparkSession.builder.appName('DMXPredictiveAnalytics')\
        .config("spark.jars", "/home/fidel/cache_pretrained/sparknlpFATjar.jar")\
        .master('local[*]').getOrCreate()
sparkTest.sparkContext.setLogLevel('ERROR')

class TextAnalysisFirstTestCase():
    spark: None

    def __init__(self):
        pass

    def textAnalytics(self, infoData):
        sparkSession = infoData.get(pc.SPARK)
        global spark
        spark = sparkSession

        datasetPath = infoData.get(pc.DATASETPATH)
        try:
            dataset = spark.read.parquet(datasetPath)
        except:
            dataset = spark.read.csv(datasetPath, header=True)
        dataset = pu.addInternalId(dataset)
        infoData.update({pc.DATASET: dataset})
        # below method textPreprocessing is related to sentiment analysis
        # make sure you will make it common for both sentiment as well text analtics.
        dataset = self.textPreProcessing(infoData)
        ''' 
        after that try to use the tf-idf method for finding frequencies and all.
        '''
        clusteredDataset = self.calTFIDF(dataset, pc.DMXLEMMATIZED)

    def textPreProcessing(self, sentimentInfoData):
        sentimentColName = sentimentInfoData.get(pc.SENTIMENTCOLNAME)
        dataset = sentimentInfoData.get(pc.DATASET)
        lemmatizedModelPath = sentimentInfoData.get(pc.LEMMATIZEDPRETRAINEDMODEL)

        textProcessing = TextProcessing(sparkSession=spark)
        dataset = textProcessing.toStringDatatype(dataset, sentimentColName)
        dataset = textProcessing.replaceSpecialChar(dataset, sentimentColName)
        dataset = textProcessing.createToken(dataset, sentimentColName)
        dataset = textProcessing.stopWordsRemover(dataset, pc.DMXTOKENIZED)
        dataset = textProcessing.sparkLemmatizer(dataset, pc.DMXSTOPWORDS, lemmatizedModelPath)

        return dataset

    def calTFIDF(self, dataset, colName):
        cv = CountVectorizer(inputCol=colName, outputCol="rawFeatures")
        cvmodel = cv.fit(dataset)
        featurizedData = cvmodel.transform(dataset)

        vocab = cvmodel.vocabulary
        vocab_broadcast = sparkTest.sparkContext.broadcast(vocab)

        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idfModel = idf.fit(featurizedData)
        rescaledData = idfModel.transform(featurizedData)  # TFIDF
        return self.clusteredData(rescaledData, cvmodel)

    def clusteredData(self, dataset, cvModel):
        lda = LDA(k=20, seed=123, optimizer="em", featuresCol="features")
        ldamodel = lda.fit(dataset)

        # model.isDistributed()
        # model.vocabSize()

        ldaTopics = ldamodel.describeTopics()
        self.getTheMapping(ldaTopics, cvModel)

        '''
        topn_words = 10
        num_topics = 10
        
        topics = ldamodel.topicsMatrix().toArray()
        for topic in range(num_topics):
            print("Topic " + str(topic) + ":")
            for word in range(0, topn_words): 
                print(" " + str(topics[word][topic]))
        '''

    def getTheMapping(self, ldaTopics, model):
        def termsIdx2Term(vocabulary):
            def termsIdx2Term(termIndices):
                return [vocabulary[int(index)] for index in termIndices]

            return udf(termsIdx2Term, ArrayType(StringType()))

        #this model is of countVectorizer not the ldamodel.
        vocabList = model.vocabulary
        final = ldaTopics.withColumn("Terms", termsIdx2Term(vocabList)("termIndices"))
        return final

if (__name__ == "__main__"):
    # datasetPath = "/dev/dmxdeepinsight/datasets/SAHIL_HAEIRENGLISHONLY.parquet"
    # datasetPath = "/dev/dmxdeepinsight/datasets/SAHIL_HAEIRSMALDATASET.parquet"
    datasetPath = "/dev/dmxdeepinsight/datasets/movieReviewSmallDataset.csv"
    # datasetPath = "/dev/dmxdeepinsight/datasets/SAHIL_SENTIMENTFINALTESTCASE.parquet"
    # colName = "tweet"
    colName = "SentimentText"
    lemmatizedModelPath = "/dev/dmxdeepinsight/models/lemmatizedPretrainedModel"

    infoData = {
        pc.DATASETPATH: datasetPath,
        pc.SENTIMENTCOLNAME: colName,
        pc.SPARK: sparkTest,
        pc.LEMMATIZEDPRETRAINEDMODEL: lemmatizedModelPath
    }

    textAnalysis = TextAnalysisFirstTestCase()
    textAnalysis.textAnalytics(infoData)
    sparkTest.stop()