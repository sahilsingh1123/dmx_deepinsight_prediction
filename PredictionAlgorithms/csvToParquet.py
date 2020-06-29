from pyspark.sql import SparkSession
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities



spark = \
    SparkSession.builder.appName('DMXPredictiveAnalytics').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

reviewDatasetPath = "/home/fidel/Documents/knimeTestData.csv"
# reviewDatasetPath = "/home/fidel/Documents/knimeTrainingData.csv"
colsName = ["Text", "Document Class", "Prediction (Document Class)"]
# colsName = ["Text", "Document Class"]


class CsvToParquet():
    def __init__(self):
        pass

    def csvToParquet(self):
        dataset = spark.read.csv(reviewDatasetPath, header=True)
        dataset = dataset.select(colsName)  #according to the requirement
        dataset = dataset.withColumnRenamed("Document Class", "Sentiment")
        dataset = dataset.withColumnRenamed("Prediction (Document Class)", "prediction_knime")
        PredictiveUtilities.writeToParquet("knimeTestDataset","/home/fidel/Documents/","",dataset)
        # dataset = dataset.select(colsName)
        # dataset = dataset.withColumnRenamed("Document Class", "Sentiment")
        # PredictiveUtilities.writeToParquet("KNIMETRAININGDATASET","/home/fidel/Documents/","",dataset)


if(__name__=="__main__"):
    csvWriter = CsvToParquet()
    csvWriter.csvToParquet()
