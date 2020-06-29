from pyspark.sql.functions import lit, when, col

from PredictionAlgorithms.SentimentAnalysis.SentimentAnalysis import SentimentAnalysis
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu
import pandas as pd


class LexiconSA(SentimentAnalysis):
    global spark, predictionCol

    def sentimentAnalysis(self, infoData):
        infoData = self.prepareData(infoData)
        sparkSession = infoData.get(pc.SPARK)
        predictionCol = infoData.get(pc.PREDICTIONCOLM)
        self.spark = sparkSession
        self.predictionCol = predictionCol
        '''above info have original and duplicate dataset 
        now you can apply text processing in duplicate 
        dataset and perform sentiment analysis on it.'''
        dataset = self.textPreProcessing(infoData)
        lexiconDictionary = self.mergePosNegDataset(infoData)
        dataset = self.addTag(dataset, pc.DMXLEMMATIZED, lexiconDictionary)
        infoData.update({pc.DATASET: dataset})
        #now make the dataset suitable for the sentiment result dataset
        # merge the dataset with original dataset... then we will proceed
        infoData = self.mergeOriginalPredictedDataset(infoData)
        self.createResult(infoData)

        '''send back the info to the server end'''
        isCumulative = infoData.get(pc.ISCUMULATIVE)
        if isCumulative:
            infoData.pop(pc.DMX_TIMESTAMPDATASET)

        infoData.pop(pc.DATASET)
        infoData.pop(pc.ORIGINALDATASET)
        infoData.pop(pc.SPARK)
        return infoData


    '''additional functionality will be written later after this.'''

    def createTaggedDataset(self, dataset, indexList, taggedRowList, positiveNum, negativeNum, totalNum,
                            sentimentScores):
        zipData = zip(indexList, taggedRowList, positiveNum, negativeNum, totalNum, sentimentScores)
        columnList = [pc.DMXINDEX, pc.DMXTAGGEDCOLM, pc.POSITIVENUM,
                      pc.NEGATIVENUM, pc.TOTALWORDS, pc.SENTIMENTSCORE]
        pandasDataframe = pd.DataFrame(zipData, columns=columnList)
        taggedDataset = self.spark.createDataFrame(pandasDataframe)
        dataset = pu.joinDataset(dataset, taggedDataset, pc.DMXINDEX)
        '''RN not dropping the neutral sentiment-- after discussion we will decide whether to drop or not'''
        #dataset = self.dropNeutral(dataset)
        dataset = self.performSentimentAnalysis(dataset)

        return dataset

    def dropNeutral(self, dataset):
        positiveColm = pc.POSITIVENUM
        negativeColm = pc.NEGATIVENUM
        query = "!(" + positiveColm + " == 0 AND " + negativeColm + " == 0)"
        dataset = dataset.filter(query)
        return dataset


    def addTag(self,dataset,colName,sentimentDictionary):
        taggedRowList = []
        indexList = []
        positiveNum = []
        negativeNum = []
        totalNum = []
        sentimentScores = []
        for index, row in enumerate(dataset.select(pc.DMXINDEX, colName).rdd.toLocalIterator()):
            dmxIndex = row[0]
            rowList = row[1]
            sentimentData = self.createSentiment(rowList, sentimentDictionary)

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
        dataset = self.createTaggedDataset(dataset, indexList, taggedRowList, positiveNum, negativeNum, totalNum,
                                           sentimentScores)

        return dataset

    def createSentiment(self, sentimentRow, posNegDataset):
        isRowUpdate = False
        posNum = 0
        negNum = 0
        totalWords = len(sentimentRow)
        for posNegIndex, row in enumerate(
                posNegDataset.select(pc.DMXINDEX, pc.DMXDICTIONARYCOLNAME, pc.DMXSENTIMENT).rdd.toLocalIterator()):
            compareText = row[pc.DMXDICTIONARYCOLNAME]  # need to get the colm Name from user
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

    def performSentimentAnalysis(self, dataset):
        sentimentScoreMean = float(list(dataset.select(pc.SENTIMENTSCORE)
                                        .summary("mean").toPandas()[pc.SENTIMENTSCORE])[0])
        print("sentiment-mean:- ", sentimentScoreMean)
        dataset = dataset.withColumn(self.predictionCol, when(col(pc.SENTIMENTSCORE) > sentimentScoreMean,
                                                               pc.POSITIVE).otherwise(pc.NEGATIVE))
        # dataset.groupby(pc.SENTIMENTVALUE).count().show()  # just for testing purpose only.
        #select only predicted and index colm..
        dataset = dataset.select(pc.DMXINDEX, self.predictionCol)
        return dataset

    '''this is for merging of both the positive and negative dictionary/dataset together'''
    def mergePosNegDataset(self, infoData):
        positiveDatasetPath = infoData.get(pc.POSITIVEDATASETPATH)
        negativeDatasetPath = infoData.get(pc.NEGATIVEDATASETPATH)
        spark = infoData.get(pc.SPARK)

        '''positive dictionary dataset--------- call convert to string method from here'''
        positiveDataset = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(positiveDatasetPath)
        positiveDataset = positiveDataset.select(positiveDataset.columns[:1])
        positiveColName = str((positiveDataset.schema.names)[0])
        positiveDataset = positiveDataset.withColumnRenamed(positiveColName, pc.DMXDICTIONARYCOLNAME)
        positiveDataset = positiveDataset.withColumn(pc.DMXSENTIMENT, lit(pc.DMXPOSITIVE)) \
            .select(pc.DMXDICTIONARYCOLNAME, pc.DMXSENTIMENT)

        '''negative dictionary dataset'''
        negativeDataset = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load(negativeDatasetPath)
        negativeDataset = negativeDataset.select(negativeDataset.columns[:1])
        negativeColName = str((negativeDataset.schema.names)[0])
        negativeDataset = negativeDataset.withColumnRenamed(negativeColName, pc.DMXDICTIONARYCOLNAME)
        negativeDataset = negativeDataset.withColumn(pc.DMXSENTIMENT, lit(pc.DMXNEGATIVE)) \
            .select(pc.DMXDICTIONARYCOLNAME, pc.DMXSENTIMENT)

        '''before appending negative dataset with positive make all colm name similar with each other'''
        posNegDataset = positiveDataset.union(negativeDataset)
        posNegDataset = pu.addInternalId(posNegDataset)

        return posNegDataset

