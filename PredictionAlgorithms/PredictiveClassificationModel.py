from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.sql import SparkSession

from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities

spark = \
    SparkSession.builder.appName('predictive_Analysis').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')


class PredictiveClassificationModel():
    def __init__(self, trainDataRatio, dataset_add, feature_colm, label_colm, relation_list,
                 relation, userId, locationAddress, algoName, spark):
        self.trainDataRatio = trainDataRatio
        self.datasetAdd = dataset_add
        self.featuresColmList = feature_colm
        self.labelColmList = label_colm
        self.relationshipList = relation_list
        self.relation = relation
        self.userId = userId
        self.locationAddress = locationAddress
        self.algoName = algoName
        self.spark = spark

        # only for etlpart of the dataset
        self.predictiveUtilitiesObj = PredictiveUtilities()

        ETLOnDatasetStats = \
            self.predictiveUtilitiesObj.ETLOnDataset(datasetAdd=self.datasetAdd,
                                                     featuresColmList=self.featuresColmList,
                                                     labelColmList=self.labelColmList,
                                                     relationshipList=self.relationshipList,
                                                     relation=self.relation,
                                                     trainDataRatio=self.trainDataRatio,
                                                     spark=spark,
                                                     userId=userId)
        self.dataset = ETLOnDatasetStats.get("dataset")
        self.featuresColm = ETLOnDatasetStats.get("featuresColm")
        self.labelColm = ETLOnDatasetStats.get("labelColm")
        self.trainData = ETLOnDatasetStats.get("trainData")
        self.testData = ETLOnDatasetStats.get("testData")
        self.idNameFeaturesOrdered = ETLOnDatasetStats.get("idNameFeaturesOrdered")

    def classificationModelStat(self, classifier):
        trainingSummary = classifier.summary

    def logisticRegression(self):
        # family = auto,multinomial and bionomial
        logisticRegressionModelFit = \
            LogisticRegression(featuresCol=self.featuresColm, labelCol=self.labelColm,
                               maxIter=5, regParam=0.1, elasticNetParam=1.0,
                               threshold=0.3, family="auto")
        classifier = logisticRegressionModelFit.fit(self.trainData)

    def randomForestClassifierModel(self):
        randomForestClassifierModelFit = \
            RandomForestClassifier(labelCol=self.labelColm,
                                   featuresCol=self.featuresColm,
                                   numTrees=10)
        classifier = randomForestClassifierModelFit.fit(self.trainData)
