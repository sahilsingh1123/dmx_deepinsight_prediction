'''creating a sentiment analysis test class for testing purpose.'''
import time

from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import regexp_replace, Column, col, udf, array_contains, posexplode_outer, lit

from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities
from PredictionAlgorithms.SentimentAnalysis.StopWords import StopWords
from PredictionAlgorithms.SentimentAnalysis.TextProcessing import TextProcessing
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover
import pandas as pd
# import modin as pd

spark = \
    SparkSession.builder.appName('DMXPredictiveAnalytics').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# reviewDatasetPath = "/home/fidel/Documents/mergeNew.csv"
# reviewDatasetPath = "/home/fidel/Documents/rohitSirFile.csv"
reviewDatasetPath = "/home/fidel/Documents/IMDb-sample.csv"
# reviewDatasetPath = "/home/fidel/Documents/movieReviewSmallDatasetQuote.csv"
positiveDatasetPath = "/home/fidel/Documents/positiveSentiment.csv"
negativeDatasetPath = "/home/fidel/Documents/negativeSentiment.csv"


class SentimentAnalysisTest():
    dmxTokenized = "dmx_tokenized"
    dmxStopWords = "dmx_stopWords"
    dmxTaggedColm = "dmx_taggedColm"
    dmxPositive = "dmx_positive"
    dmxNegative = "dmx_negative"
    dmxSentiment = "dmx_sentiment"

    """ask the user to provide on unique value colm or indexing with all the three dataset."""
    sentimentColName = "SentimentText"
    positiveColName = "words"
    negativeColName = "words"

    def __init__(self):
        pass

    def toStringDataType(self, dataset, colName):
        datasetSchema = dataset.schema
        for schemaVal in datasetSchema:
            if (str(schemaVal.dataType) == "TimestampType"
                    or str(schemaVal.dataType) == "DateType"
                    or str(schemaVal.dataType) == "BooleanType"
                    or str(schemaVal.dataType) == "BinaryType"
                    or str(schemaVal.dataType) == "DoubleType"
                    or str(schemaVal.dataType) == "IntegerType"):
                if (schemaVal.name == colName):
                    dataset = dataset.withColumn(colName, dataset[colName].cast(StringType()))
        return dataset

    def replaceSpecialChar(self, dataset, colName):
        regexIgnore = "[^a-zA-Z ]"
        regexInclude = "[,.]"
        dataset = dataset.withColumn(colName, regexp_replace(col(colName), regexIgnore, ""))
        dataset = dataset.withColumn(colName, regexp_replace(col(colName), regexInclude, " "))

        dataset = self.createToken(dataset, colName)
        return dataset

    '''this method create token for each word and converts to lowercase'''

    def createToken(self, dataset, colName):
        sentimentTokenizer = RegexTokenizer(inputCol=colName, outputCol=self.dmxTokenized,
                                            toLowercase=True,
                                            pattern="\\W")  # update the constant with in predictive constant class.
        dataset = sentimentTokenizer.transform(dataset)

        dataset = self.stopWordsRemover(dataset, self.dmxTokenized)
        return dataset

    def stopWordsRemover(self, dataset, colName):
        stopWordsList = StopWords.stopWordsKNIME
        sentimentStopWordRemover = StopWordsRemover(inputCol=colName, outputCol=self.dmxStopWords,stopWords=stopWordsList)
        dataset = sentimentStopWordRemover.transform(dataset)
        textProcessing = TextProcessing(sparkSession=spark)
        dataset = textProcessing.stemming(dataset, pc.DMXSTOPWORDS)
        dataset = textProcessing.ngrams(dataset,pc.DMXSTOPWORDS,2)
        dataset = textProcessing.lemmatization(dataset,pc.DMXSTOPWORDS)
        return dataset

    def getSentimentInfo(self, sentimentRow):
        posNum = 0
        negNum = 0
        totalWords = len(sentimentRow)
        for index, value in enumerate(sentimentRow):
            # print("--------------inside sentiment score loop--------------------")
            if (value.endswith("[dmx_positive]")):  # take sentiment name from user end.
                posNum = posNum + 1
            if (value.endswith("[dmx_negative")):
                negNum = negNum + 1

        sentimentScore = round((posNum - negNum) / totalWords, 4)
        sentimentInfo = {
            pc.POSITIVENUM: posNum,
            pc.NEGATIVENUM: negNum,
            pc.TOTALWORDS: totalWords,
            pc.SENTIMENTSCORE: sentimentScore
        }

        return sentimentInfo

    def addTag(self, sentimentRow, posNegDataset):
        isRowUpdate = False
        posNum = 0
        negNum = 0
        totalWords = len(sentimentRow)
        for posNegIndex, row in enumerate(
                posNegDataset.select(pc.DMXINDEX, "words", pc.DMXSENTIMENT).rdd.toLocalIterator()):
            compareText = row["words"]  # need to get the colm Name from user
            sentiment = row[pc.DMXSENTIMENT]
            if (sentimentRow.__contains__(compareText)):
                isRowUpdate = True
                for index, text in enumerate(sentimentRow):
                    if (compareText.__eq__(text)):
                        if (sentiment.__eq__(pc.DMXPOSITIVE)):
                            posNum = posNum + 1
                        elif (sentiment.__eq__(pc.DMXNEGATIVE)):
                            negNum = negNum + 1
                        sentimentRow[index] = text + "[" + sentiment + "]"

        # calculate the no of positive words, negative words, and total number of words, and sentiment score.
        if (isRowUpdate):
            sentimentScore = round((posNum - negNum) / totalWords, 4)
            # sentimentInfo = self.getSentimentInfo(sentimentRow)
        else:
            sentimentScore = 0.0

        sentimentData = {
            pc.SENTIMENTROW: sentimentRow,
            pc.POSITIVENUM: posNum,
            pc.NEGATIVENUM: negNum,
            pc.TOTALWORDS: totalWords,
            pc.SENTIMENTSCORE: sentimentScore
        }

        return sentimentData

    def createSentimentData(self, dataset, colName, sentimentDictionary):
        taggedRowList = []
        indexList = []
        positiveNum = []
        negativeNum = []
        totalNum = []
        sentimentScores = []
        for index, row in enumerate(dataset.select(pc.DMXINDEX,colName).rdd.toLocalIterator()):
            dmxIndex = row[0]
            rowList = row[1]
            sentimentData = self.addTag(rowList, sentimentDictionary)

            taggedRow = sentimentData.get(pc.SENTIMENTROW)
            posNum = sentimentData.get(pc.POSITIVENUM)
            negNum = sentimentData.get(pc.NEGATIVENUM)
            totalWords = sentimentData.get(pc.TOTALWORDS)
            sentimentScore = sentimentData.get(pc.SENTIMENTSCORE)

            indexList.append(dmxIndex)
            taggedRowList.append(taggedRow)
            positiveNum.append(posNum)
            negativeNum.append(negNum)
            totalNum.append(totalWords)
            sentimentScores.append(sentimentScore)
            print(sentimentScore)

        '''create dataset of the taggedColm -- & join the tagged dataset with the original dataset'''
        taggedDataset = self.createTaggedDataset(indexList, taggedRowList, positiveNum, negativeNum, totalNum,
                                                 sentimentScores)
        dataset = self.joinDataset(dataset, taggedDataset, pc.DMXINDEX)
        return dataset

    def createTaggedDataset(self, indexList, taggedRowList, positiveNum, negativeNum, totalNum, sentimentScores):
        zipData = zip(indexList, taggedRowList, positiveNum, negativeNum, totalNum, sentimentScores)
        columnList = [pc.DMXINDEX, self.dmxTaggedColm, pc.POSITIVENUM,
                      pc.NEGATIVENUM, pc.TOTALWORDS, pc.SENTIMENTSCORE]
        pandasDataframe = pd.DataFrame(zipData, columns=columnList)
        dataset = spark.createDataFrame(pandasDataframe)
        return dataset

    def joinDataset(self, originalDataset, taggedDataset, joinOnColumn):
        dataset = originalDataset.join(taggedDataset, on=[joinOnColumn]).sort(joinOnColumn)
        return dataset

    '''taking out neutral sentiment based on positive and negative words count'''
    def seperateNeutralData(self,dataset):
        positiveColm = pc.POSITIVENUM
        negativeColm = pc.NEGATIVENUM
        query = "!(" + positiveColm + " == 0 AND " + negativeColm + " == 0)"
        neutralQuery = "(" + positiveColm + " == 0 AND " + negativeColm + " == 0)"
        neutralDataset = dataset.filter(neutralQuery)
        dataset = dataset.filter(query)
        seperateNeutralDataInfo = {
            pc.NEUTRALDATASET: neutralDataset,
            pc.DATASET: dataset
        }
        return seperateNeutralDataInfo

    def performSentimentAnalysis(self, dataset):
        sentimentScoreMean = float(list(dataset.select(pc.SENTIMENTSCORE)
                                        .summary("mean").toPandas()[pc.SENTIMENTSCORE])[0])
        print("sentiment-mean:- ", sentimentScoreMean)
        dataset = dataset.withColumn(pc.SENTIMENTVALUE, F.when(F.col(pc.SENTIMENTSCORE) > sentimentScoreMean,
                                                               pc.POSITIVE).otherwise(pc.NEGATIVE))
        # dataset.groupby(pc.SENTIMENTVALUE).count().show()  # just for testing purpose only.

        return dataset

    def handleNeutral(self,dataset):
        dataset = dataset.withColumn(pc.SENTIMENTVALUE, lit(pc.NEUTRAL))
        return dataset

    def appendDataset(self,datasetOne,datasetTwo):
        dataset = datasetOne.unionAll(datasetTwo)
        return dataset

    def getDataset(self):
        '''review dataset'''
        start_time = time.localtime(time.time())
        movieReviewDataset = spark.read.csv(reviewDatasetPath, header=True)
        # writeToParquet = PredictiveUtilities.writeToParquet("movieDatasetParquet","/home/fidel/Documents/","",movieReviewDataset)
        # movieReviewDataset = movieReviewDataset.select(self.sentimentColName)
        movieReviewDataset.na.drop()
        '''temp test'''
        movieReviewDataset = movieReviewDataset.drop("URL")
        movieReviewDataset = movieReviewDataset.withColumnRenamed("SentimentText", "Text")
        movieReviewDataset = movieReviewDataset.withColumn(self.sentimentColName, movieReviewDataset["Text"])

        movieReviewDataset = movieReviewDataset.withColumn(
            pc.DMXINDEX, F.monotonically_increasing_id())

        # change the schema of the dataset to String type
        movieReviewDataset = self.toStringDataType(movieReviewDataset, self.sentimentColName)
        '''remove special characters -> tokenization -> stopWords Removal'''
        movieReviewDataset = self.replaceSpecialChar(movieReviewDataset, self.sentimentColName)
        movieReviewDataset.show()
        """
        make the datatype of the dictionary to string only.
        """
        '''positive dictionary dataset'''
        positiveDataset = spark.read.csv(positiveDatasetPath, header=True)
        # positiveData = PredictiveUtilities.writeToParquet("positiveSentimentDatasetParquet", "/home/fidel/Documents/", "",positiveDataset)
        positiveDataset = positiveDataset.withColumn(pc.DMXSENTIMENT, lit(pc.DMXPOSITIVE))

        '''negative dictionary dataset'''
        negativeDataset = spark.read.csv(negativeDatasetPath, header=True)
        # negData = PredictiveUtilities.writeToParquet("negativeSentimentDatasetParquet", "/home/fidel/Documents/", "",negativeDataset)
        negativeDataset = negativeDataset.withColumn(pc.DMXSENTIMENT, lit(pc.DMXNEGATIVE))

        posNegSentimentDataset = positiveDataset.unionAll(negativeDataset)
        posNegSentimentDataset = posNegSentimentDataset.withColumn(
            pc.DMXINDEX, F.monotonically_increasing_id())

        finalDataset = self.createSentimentData(movieReviewDataset, pc.DMXLEMMATIZED, posNegSentimentDataset)
        '''handling neutral sentences..'''
        seperateNeutral = self.seperateNeutralData(finalDataset)
        neutralDataset = seperateNeutral.get(pc.NEUTRALDATASET)
        nonNeutralDataset = seperateNeutral.get(pc.DATASET)

        nonNeutralDataset = self.performSentimentAnalysis(nonNeutralDataset)
        neutralDataset = self.handleNeutral(neutralDataset)

        finalDataset = self.appendDataset(nonNeutralDataset,neutralDataset).sort(pc.DMXINDEX)
        # finalDataset = self.performSentimentAnalysis(finalDataset)
        end_time = time.localtime(time.time())
        print(start_time , "\n", end_time)
        finalDataset.groupby(pc.SENTIMENTVALUE).count().show()
        # finalDataset.groupby("Sentiment Prediction").count().show()
        #write to csv
        # finalDataset.drop(pc.DMXSTEMMEDWORDS, pc.DMXSTOPWORDS, pc.DMXTOKENIZED, pc.DMXTAGGEDCOLM,pc.DMXSTEMMEDWORDS, pc.DMXNGRAMS).coalesce(
        #     1).write.mode("overwrite").format("com.databricks.spark.csv").option("header", "true").csv(
        #     "/home/fidel/Documents/knimeStopWordsRohitSirDataWithStem.csv")
        finalDataset.show()


if __name__ == '__main__':
    sentimentAnalysisTest = SentimentAnalysisTest()
    sentimentAnalysisTest.getDataset()
