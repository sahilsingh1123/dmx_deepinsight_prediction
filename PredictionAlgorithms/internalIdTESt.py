from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from pyspark.sql import SparkSession
from pyspark import Accumulator, AccumulatorParam

sparkTest = SparkSession.builder.appName('DMXPredictiveAnalytics').master('local[*]').getOrCreate()

#creating a accumulator class extending accumulator.
class accumulatorClass(AccumulatorParam):
    def zero(self, value):
        return value

    def addInPlace(self, value1, value2):
        return value1 + value2


class internalIDTest():
    def __init__(self):
        pass

    def startIndexing(self,infoData):
        datasetPath = infoData.get(pc.FILELOCATION)
        spark = infoData.get(pc.SPARK)
        noOfPartitions = 4
        dataset = spark.read.csv(datasetPath, header=True).repartition(noOfPartitions)

        # after that create accumulator and then pass it to mapPartitionwithIndex method of spark
        #then call the class which extends accumulatorParam.


        accumulator_test = spark.sparkContext.accumulator({},accumulatorClass())







if (__name__ == '__main__'):
    infoData = {
        pc.FILELOCATION: "/home/fidel/Documents/sentimentMovieReviewDataset.csv",
        pc.SPARK: sparkTest,
    }

    startIndex = internalIDTest().startIndexing(infoData)
    sparkTest.stop()
