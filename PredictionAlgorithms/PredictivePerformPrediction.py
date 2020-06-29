from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel

from PredictionAlgorithms.PredictiveConstants import PredictiveConstants
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities
import pyspark.sql.functions as F
import uuid


class PredictivePerformPrediction():
    def __init__(self):
        pass

    def prediction(self, predictiveData):

        '''creating duplicate dataset to avoid the datatype change of the original dataset '''
        datasetAdd = predictiveData.get(PredictiveConstants.DATASETADD)
        spark = predictiveData.get(PredictiveConstants.SPARK)
        dataset = spark.read.parquet(datasetAdd)

        # adding extra index column in the dataset
        dataset = PredictiveUtilities.addInternalId(dataset)
        predictiveData.update({
            PredictiveConstants.DATASET: dataset
        })

        etlStats = PredictiveUtilities.performETL(etlInfo=predictiveData)
        dataset = etlStats.get(PredictiveConstants.DATASET)
        originalDataset = etlStats.get(PredictiveConstants.ORIGINALDATASET)

        algoName = predictiveData.get(PredictiveConstants.ALGORITHMNAME)
        modelStorageLocation = predictiveData.get(PredictiveConstants.MODELSTORAGELOCATION)
        modelName = predictiveData.get(PredictiveConstants.MODELSHEETNAME)
        datasetName = predictiveData.get(PredictiveConstants.DATASETNAME)
        spark = predictiveData.get(PredictiveConstants.SPARK)
        locationAddress = predictiveData.get(PredictiveConstants.LOCATIONADDRESS)

        if PredictiveConstants.LINEAR_REG.__eq__(algoName) or \
                PredictiveConstants.RIDGE_REG.__eq__(algoName) or PredictiveConstants.LASSO_REG.__eq__(algoName):
            regressionPrediction = LinearRegressionModel.load(modelStorageLocation)
        if PredictiveConstants.RANDOMFORESTALGO.__eq__(algoName):
            regressionPrediction = RandomForestRegressionModel.load(modelStorageLocation)
        if PredictiveConstants.GRADIENTBOOSTALGO.__eq__(algoName):
            regressionPrediction = GBTRegressionModel.load(modelStorageLocation)

        dataset = dataset.drop(modelName)
        originalDataset = originalDataset.drop(modelName)
        dataset = regressionPrediction.transform(dataset)
        dataset = dataset.select(PredictiveConstants.DMXINDEX, modelName)
        finalDataset = originalDataset.join(dataset, on=[PredictiveConstants.DMXINDEX]) \
            .sort(PredictiveConstants.DMXINDEX).drop(PredictiveConstants.DMXINDEX)

        # predictionData = predictionData.drop(featuresColm)
        #
        # #dropping extra added column
        # if indexedFeatures:
        #     indexedFeatures.extend(oneHotEncodedFeaturesList)
        #     predictionData = predictionData.drop(*indexedFeatures)
        # else:
        #     predictionData = predictionData

        # overWriting the original dataset
        '''this step is needed to write because of the nature of spark to not read or write whole data at once
        it only takes limited data to memory and another problem was lazy evaluation of spark.
        so overwriting the same dataset which is already in the memory is not possible'''
        emptyUserId = ''
        randomUUID = str(uuid.uuid4())
        fileNameWithPathTemp = locationAddress + randomUUID + datasetName + "_temp.parquet" #correct the name.
        finalDataset.write.parquet(fileNameWithPathTemp, mode="overwrite")  # send this path to java for deletion
        predictionDataReadAgain = spark.read.parquet(fileNameWithPathTemp)

        predictionTableData = \
            PredictiveUtilities.writeToParquet(fileName=datasetName,
                                               locationAddress=locationAddress,
                                               userId=emptyUserId,
                                               data=predictionDataReadAgain)
        return predictionTableData
