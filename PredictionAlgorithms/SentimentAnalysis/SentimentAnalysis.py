from abc import ABC, abstractmethod
from PredictionAlgorithms.SentimentAnalysis.TextProcessing import TextProcessing
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu
import operator
import pyspark.sql.functions as F
import datetime
import pandas as pd
from pyspark.sql.types import TimestampType

class SentimentAnalysis(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sentimentAnalysis(self, infoData):
        raise NotImplementedError("subClass must implement abstract method")

    def prepareData(self, infoData):
        infoData = self.createSentimentData(infoData)
        dataset = infoData.get(pc.DATASET)
        originalDataset, duplicateDataset = self.createDuplicateDataset(dataset)
        infoData.update({pc.DATASET: duplicateDataset,
                         pc.ORIGINALDATASET: originalDataset})
        return infoData

    def textPreProcessing(self, sentimentInfoData):
        sentimentColName = sentimentInfoData.get(pc.SENTIMENTCOLNAME)
        dataset = sentimentInfoData.get(pc.DATASET)
        lemmatizedModelPath = sentimentInfoData.get(pc.LEMMATIZEDPRETRAINEDMODEL)
        spark = sentimentInfoData.get(pc.SPARK)

        textProcessing = TextProcessing(sparkSession=spark)
        dataset = textProcessing.toStringDatatype(dataset, sentimentColName)
        dataset = textProcessing.replaceSpecialChar(dataset, sentimentColName)
        dataset = textProcessing.createToken(dataset, sentimentColName)
        dataset = textProcessing.stopWordsRemover(dataset, pc.DMXTOKENIZED)
        dataset = textProcessing.sparkLemmatizer(dataset, pc.DMXSTOPWORDS, lemmatizedModelPath)

        return dataset

    '''need to create the dataset based on communication ID.'''
    def createSentimentData(self, infoData):
        sentimentColName = infoData.get(pc.SENTIMENTCOLNAME)
        dateColName = infoData.get(pc.DATECOL)
        idColName = infoData.get(pc.CONVERSATIONIDCOL)
        datasetPath = infoData.get(pc.SENTIMENTDATASETPATH)
        isCumulative = infoData.get(pc.ISCUMULATIVE)
        spark = infoData.get(pc.SPARK)
        dataset = spark.read.parquet(datasetPath)
        dataset = dataset.select(idColName, dateColName, sentimentColName)
        dataset = dataset.dropna()
        if isCumulative:
            timeStampDataset = self.getTimeStamp(infoData, dataset)
            infoData.update({pc.DMX_TIMESTAMPDATASET: timeStampDataset})
            dataset = self.mergeDataOnIdWithTime(dataset, infoData)
        infoData.update({pc.DATASET: dataset})
        return infoData

    def getTimeStamp(self, infoData, dataset):
        spark = infoData.get(pc.SPARK)
        dateCol = infoData.get(pc.DATECOL)
        convIdCol = infoData.get(pc.CONVERSATIONIDCOL)
        convIdInfo = {}
        dataset.createOrReplaceTempView("sentimentDataset")
        dataset = spark.sql("select * from sentimentDataset order by ({convId}, {date})".format(convId = convIdCol, date = dateCol))
        for index, value in enumerate(dataset.rdd.toLocalIterator()):
            dateDict = {}
            convId = value[convIdCol]
            dateVal = value[dateCol]
            if convId in convIdInfo:
                convIdInfo.get(convId).update({pc.DMX_ENDTIME: str(dateVal)})
            else:
                dateDict.update({
                    pc.DMX_STARTTIME: str(dateVal),
                    pc.DMX_ENDTIME: str(dateVal)
                })
                convIdInfo.update({
                    convId: dateDict
                })
        startTimeList = []
        endTimeList = []
        convIdList = list(convIdInfo.keys())
        for key in convIdList:
            startTimeList.append(convIdInfo.get(key).get(pc.DMX_STARTTIME))
            endTimeList.append(convIdInfo.get(key).get(pc.DMX_ENDTIME))
        dataset = self.createTimeStampDataset(convIdList, startTimeList, endTimeList, convIdCol, spark)

        return dataset

    def createTimeStampDataset(self,convIdList, startTimeList, endTimeList, convIdCol, spark):
        dataList = zip(convIdList,startTimeList,endTimeList)
        columnsList = [convIdCol, pc.DMX_STARTTIME, pc.DMX_ENDTIME]
        pdDataframe = pd.DataFrame(dataList, columns=columnsList)
        timestampDataset = spark.createDataFrame(pdDataframe)
        timestampDataset = timestampDataset.withColumn(pc.DMX_STARTTIME, timestampDataset[pc.DMX_STARTTIME].cast(TimestampType()))
        timestampDataset = timestampDataset.withColumn(pc.DMX_ENDTIME, timestampDataset[pc.DMX_ENDTIME].cast(TimestampType()))

        return timestampDataset

    def mergeDataOnId(self, dataset, infoData):
        spark = infoData.get(pc.SPARK)
        convIdCol = infoData.get(pc.CONVERSATIONID)
        timeStampCol = infoData.get(pc.TIMESTAMP)
        sentimentCol = infoData.get(pc.SENTIMENTCOLNAME)
        dataset = dataset.select(convIdCol, timeStampCol, sentimentCol)
        dataset.createOrReplaceTempView("sentimentDataset")
        query = "select {convId}, concat_ws(' ', collect_list({sentCol})) as {sentCol} from sentimentDataset group by {convId}"\
            .format(convId = convIdCol, sentCol = sentimentCol)
        dataset = spark.sql(query)
        return dataset

    def mergeDataOnIdWithTime(self, dataset, infoData):
        convIdCol = infoData.get(pc.CONVERSATIONIDCOL)
        dateCol = infoData.get(pc.DATECOL)
        sentimentCol = infoData.get(pc.SENTIMENTCOLNAME)
        dataset = dataset.select(convIdCol, dateCol, sentimentCol)
        dataset = dataset.groupby(convIdCol) \
            .agg(F.collect_list(F.struct(dateCol, sentimentCol)).alias(sentimentCol))
        def sorter(l):
            res = sorted(l, key=operator.itemgetter(0))
            return [item[1] for item in res]

        sort_udf = F.udf(sorter)
        dataset = dataset.select(convIdCol, sort_udf(sentimentCol).alias(sentimentCol))
        #in case u need to remove the list symbol from the text column
        #df.withColumn("test_123", concat_ws(",", "test_123")).show()

        return dataset

    def addColToDataset(self, dataset, datasetName, modelName):
        timeStamp = datetime.datetime.now()
        dataset = dataset.withColumn(pc.RESULTDATASETCOLNAME, F.lit(datasetName))
        dataset = dataset.withColumn(pc.DMX_CREATEDON, F.lit(timeStamp))
        dataset = dataset.withColumn(pc.MODELNAMECOL, F.lit(modelName))
        return dataset

    def createDuplicateDataset(self, dataset):
        originalDataset, duplicateDataset = pu.duplicateDataset(dataset)
        return originalDataset, duplicateDataset

    def getTimeStampForNonCumulative(self, infoData):
        dataset = infoData.get(pc.ORIGINALDATASET)
        dateCol = infoData.get(pc.DATECOL)
        dataset = dataset.withColumnRenamed(dateCol, pc.DMX_STARTTIME)
        dataset = dataset.withColumn(pc.DMX_ENDTIME, dataset[pc.DMX_STARTTIME])
        return dataset

    def mergeNonCumTimeStampCol(self, infoData):
        datasetName = infoData.get(pc.DATASETNAME)
        modelName = infoData.get(pc.MODELNAME)
        dataset = self.getTimeStampForNonCumulative(infoData)
        dataset = self.addColToDataset(dataset,datasetName, modelName)
        return dataset

    def createResult(self, infoData):
        isCumulative = infoData.get(pc.ISCUMULATIVE)
        if isCumulative:
            dataset = self.mergeTimeStampCol(infoData)
        else:
            dataset = self.mergeNonCumTimeStampCol(infoData)
        self.writeSentimentResult(infoData,dataset)

    def mergeTimeStampCol(self, infoData):
        dataset = infoData.get(pc.ORIGINALDATASET)
        modelName = infoData.get(pc.MODELNAME)
        datasetName = infoData.get(pc.DATASETNAME)
        timestampDataset = infoData.get(pc.DMX_TIMESTAMPDATASET)
        convIdCol = infoData.get(pc.CONVERSATIONIDCOL)
        dataset = pu.joinDataset(dataset, timestampDataset, convIdCol)
        dataset = self.addColToDataset(dataset, datasetName, modelName)
        return dataset

    def writeSentimentResult(self, infoData, dataset):
        spark = infoData.get(pc.SPARK)
        resultDatasetPath = infoData.get(pc.SENTIMENTRESULTDATASETPATH)
        resultDatasetPathTemp = resultDatasetPath.replace(".parquet", "_TEMP.parquet")
        resultDataset = spark.read.parquet(resultDatasetPath)
        resultDataset = resultDataset.drop(pc.DMX_INTERNAL_ID)
        dataset = self.makeDataFitForResult(infoData,dataset)
        resultDataset = resultDataset.union(dataset)

        resultDataset.write.parquet(resultDatasetPathTemp, mode="overwrite")
        resultDatasetFinal = spark.read.parquet(resultDatasetPathTemp)
        resultDatasetFinal.write.parquet(resultDatasetPath, mode="overwrite")

    def makeDataFitForResult(self, infoData,dataset):
        predictionColName = infoData.get(pc.PREDICTIONCOLM)
        convIdCol = infoData.get(pc.CONVERSATIONIDCOL)
        dataset = dataset.withColumnRenamed(predictionColName, pc.RESULTCOLNAME)
        dataset = dataset.withColumnRenamed(convIdCol, pc.CONVERSATIONID)
        dataset = dataset.select(pc.RESULTDATASETCOLNAME,pc.DMX_STARTTIME, pc.DMX_ENDTIME,pc.MODELNAMECOL,
                                 pc.DMX_CREATEDON, pc.RESULTCOLNAME, pc.CONVERSATIONID)
        return dataset

    def mergeOriginalPredictedDataset(self, infoData):
        originalDataset = infoData.get(pc.ORIGINALDATASET)
        dataset = infoData.get(pc.DATASET)
        originalDataset = pu.joinDataset(originalDataset, dataset, pc.DMXINDEX)
        infoData.update({pc.ORIGINALDATASET: originalDataset})
        return infoData




