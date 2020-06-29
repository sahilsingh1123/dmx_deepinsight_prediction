from PredictionAlgorithms.SentimentAnalysis.SparkNLP import SparkNLP
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu
from PredictionAlgorithms.SentimentAnalysis.SentimentAnalysis import SentimentAnalysis
from pyspark.sql.functions import array_join
from sparknlp.pretrained import ViveknSentimentModel
from pyspark.sql.functions import when

class ViveknPretrainedModel(SentimentAnalysis):
    def sentimentAnalysis(self, infoData):
        infoData = self.prepareData(infoData)
        # originalDataset = infoData.get(pc.ORIGINALDATASET)
        infoData = SparkNLP.textProcessing(infoData)
        dataset = self.vivekSentimentPretrained(infoData)
        """before merging original dataset and writing get the dataset in proper format."""
        '''merge the original dataset and predicted dataset together based on the dmxIndex'''
        # originalDataset = pu.joinDataset(originalDataset, dataset, pc.DMXINDEX)
        # infoData.update({pc.ORIGINALDATASET: originalDataset})
        infoData.update({pc.DATASET: dataset})
        infoData = self.mergeOriginalPredictedDataset(infoData)
        self.createResult(infoData)

        """check if isCumulative is true or false and accordingly take the methods..."""
        # datasetInfo = self.writeDataset(originalDataset, infoData)
        isCumulative = infoData.get(pc.ISCUMULATIVE)
        if isCumulative:
            infoData.pop(pc.DMX_TIMESTAMPDATASET)

        infoData.pop(pc.DATASET)
        infoData.pop(pc.ORIGINALDATASET)
        infoData.pop(pc.SPARK)
        return infoData

    def writeDataset(self, dataset, infoData):
        storageLocation = infoData.get(pc.STORAGELOCATION)
        modelName = infoData.get(pc.MODELNAME)
        userId = infoData.get(pc.USERID)

        """
        write the dataset if not exists and if do then append the new data inside the dataset
        - keep the datasetID information, and coversationID should be unique in the dataset.
        """
        datasetInfo = pu.writeToParquet(modelName, storageLocation, userId, dataset)
        return datasetInfo

    """
    directly predict the sentiment without any need of training the dataset or having the label colm
    """

    def vivekSentimentPretrained(self, infoData):
        dataset = infoData.get(pc.DATASET)
        viveknPretrainedModelPath = infoData.get(pc.VIVEKNPRETRAINEDMODEL)
        predictionCol = infoData.get(pc.PREDICTIONCOLM)
        """use to download it once later we need to load it from the local to avoid dependency on online downloader."""
        viveknSentiment = ViveknSentimentModel.load(viveknPretrainedModelPath).setInputCols(
            ["document", pc.DMXSTOPWORDS]).setOutputCol(predictionCol)
        dataset = viveknSentiment.transform(dataset)
        dataset = dataset.withColumn(predictionCol, array_join(predictionCol + ".result", ""))
        dataset = dataset.select(pc.DMXINDEX, predictionCol)
        dataset = dataset.withColumn(predictionCol, when(dataset[predictionCol] == "negative", pc.NEGATIVE)
                                     .when(dataset[predictionCol] == "positive", pc.POSITIVE).otherwise(pc.NEUTRAL))
        return dataset
