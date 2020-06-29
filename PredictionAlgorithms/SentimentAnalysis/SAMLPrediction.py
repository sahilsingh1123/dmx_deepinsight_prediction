import json
from PredictionAlgorithms.SentimentAnalysis.TextProcessing import TextProcessing
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu
from pyspark.ml.classification import GBTClassificationModel, DecisionTreeClassificationModel
from pyspark.ml.feature import CountVectorizerModel
from pyspark.sql import SparkSession

'''sahil- fix this'''
sparkTest = \
    SparkSession.builder.appName('DMXPredictiveAnalytics').master('local[*]').getOrCreate()
sparkTest.sparkContext.setLogLevel('ERROR')


class SAMLPrediction(object):
    def __init__(self):
        pass

    def prediction(self, infoData):
        isNgram = False if infoData.get(pc.ISNGRAM) == None else infoData.get(pc.ISNGRAM)
        predictionColm = infoData.get(pc.PREDICTIONCOLM)
        algoName = infoData.get(pc.ALGORITHMNAME)
        modelStorageLocation = infoData.get(pc.MODELSTORAGELOCATION)
        spark = infoData.get(pc.SPARK)
        datasetPath = infoData.get(pc.SENTIMENTDATASETPATH)
        originalDataset = spark.read.parquet(datasetPath)
        originalDataset = pu.addInternalId(originalDataset)
        infoData.update({pc.DATASET: originalDataset})

        infoData = self.dataTransformation(infoData)

        dataset = infoData.get(pc.DATASET)
        if (isNgram):
            """sahil-- handle the none value for ngram parameter at the time of data creation"""
            textProcessing = TextProcessing(sparkSession=spark)
            ngramPara = infoData.get(pc.NGRAMPARA)
            dataset = textProcessing.ngrams(dataset, pc.DMXLEMMATIZED, ngramPara)

        """
        -- sahil- hardCoding the algorithm name for comparision handle this while finalising
        """
        if ("GradientBoostClassifier".__eq__(algoName)):
            predictionModel = GBTClassificationModel.load(modelStorageLocation)
        if ("DecisionTreeClassifier".__eq__(algoName)):
            predictionModel = DecisionTreeClassificationModel.load(modelStorageLocation)

        dataset = dataset.drop(predictionColm)
        originalDataset = originalDataset.drop(predictionColm)
        dataset = predictionModel.transform(dataset)
        """calling indexToString method after the prediction"""
        infoData.update({pc.DATASET: dataset})
        infoData = self.invertIndex(infoData)

        dataset = infoData.get(pc.DATASET)
        dataset = dataset.select(pc.DMXINDEX, predictionColm)
        finalDataset = pu.joinDataset(originalDataset, dataset, pc.DMXINDEX)
        return finalDataset

    def dataTransformation(self, infoData):
        """
        :returns processed data with cleaning,stopWords removal, special character removal
        :argument sparkInstance, sentimentColName, datasetPath
        :rtype spark dataframe
        """
        dataset = self.textPreProcessing(infoData)
        infoData.update({pc.DATASET: dataset})
        infoData = self.countVectorizer(infoData)
        return infoData

    """
    using countvectorizer in case of sentiment analysis-- cannot use onehOtEncoder for array of String
    """

    def countVectorizer(self, infoData):
        originalColName = infoData.get(pc.ORIGINALCOLMNAME)
        dataset = infoData.get(pc.DATASET)
        oneHotEncoderMapping = infoData.get(pc.ONEHOTENCODERPATHMAPPING)
        countVectorizerPath = oneHotEncoderMapping.get(originalColName)
        countVectorizer = CountVectorizerModel.load(countVectorizerPath)
        encodedColmName = infoData.get(pc.ENCODEDCOLM)
        dataset = dataset.drop(encodedColmName)

        dataset = countVectorizer.transform(dataset)
        infoData.update({pc.DATASET: dataset})

        infoData = pu.featureAssembler(infoData)

        return infoData

    def textPreProcessing(self, sentimentInfoData):
        sentimentColName = sentimentInfoData.get(pc.SENTIMENTCOLNAME)
        spark = sentimentInfoData.get(pc.SPARK)
        dataset = sentimentInfoData.get(pc.DATASET)

        textProcessing = TextProcessing(sparkSession=spark)
        dataset = textProcessing.toStringDatatype(dataset, sentimentColName)
        dataset = textProcessing.replaceSpecialChar(dataset, sentimentColName)
        dataset = textProcessing.createToken(dataset, sentimentColName)
        dataset = textProcessing.stopWordsRemover(dataset, pc.DMXTOKENIZED)
        dataset = textProcessing.lemmatization(dataset, pc.DMXSTOPWORDS)

        return dataset

    def invertIndex(self, infoData):
        originalColName = infoData.get(pc.ORIGINALCOLMNAME)
        indexerPath = (infoData.get(pc.INDEXERPATHMAPPING)).get(originalColName)
        infoData.update({pc.INDEXERPATH: indexerPath})
        dataset = pu.indexToString(infoData)
        infoData.update({pc.DATASET: dataset})

        """
        datasetTest = datasetTest.select("Text","Sentiment","prediction_knime", predictionColm)
        datasetTest.coalesce(
            1).write.mode("overwrite").format("com.databricks.spark.csv").option("header", "true").csv(
            "/home/fidel/Documents/decisionTreeKNIMEPrediction.csv")
        """
        return infoData


if (__name__ == "__main__"):
    infoData = json.load(open("/home/fidel/Documents/movieReviewDecisionTree.json"))
    datasetPath = "/home/fidel/Documents/KNIMETESTDATASET.parquet"
    infoData.update({pc.SPARK: sparkTest,
                     pc.SENTIMENTDATASETPATH: datasetPath})
    sentimentPrediction = SAMLPrediction()
    sentimentPrediction.prediction(infoData)
