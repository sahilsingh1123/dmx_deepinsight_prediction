import json
import re
from scipy.stats import norm
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import abs as absSpark, sqrt as sqrtSpark, mean as meanSpark, stddev as stddevSpark, lit
from pyspark.sql.types import *
from PredictionAlgorithms.PredictiveDataTransformation import PredictiveDataTransformation
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from pyspark.ml.feature import CountVectorizer, VectorAssembler, StringIndexer, StringIndexerModel, IndexToString

class PredictiveUtilities():

    @staticmethod
    def ETLOnDataset(datasetAdd, featuresColmList, labelColmList,
                     relationshipList, relation, trainDataRatio, spark, userId):

        dataset = spark.read.parquet(datasetAdd)
        # changing the relationship of the colm
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataset = \
            dataTransformationObj.colmTransformation(
                colmTransformationList=relationshipList) if relation == pc.NON_LINEAR else dataset
        # transformation
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataTransformationResult = dataTransformationObj.dataTranform(datasetAdd) #tempFix
        dataset = dataTransformationResult[pc.DATASET]
        categoricalFeatures = dataTransformationResult.get(pc.CATEGORICALFEATURES)
        numericalFeatures = dataTransformationResult.get(pc.NUMERICALFEATURES)
        maxCategories = dataTransformationResult.get(pc.MAXCATEGORIES)
        categoryColmStats = dataTransformationResult.get(pc.CATEGORYCOLMSTATS)
        indexedFeatures = dataTransformationResult.get(pc.INDEXEDFEATURES)
        idNameFeaturesOrdered = dataTransformationResult.get(pc.IDNAMEFEATURESORDERED)
        oneHotEncodedFeaturesList = dataTransformationResult.get(pc.ONEHOTENCODEDFEATURESLIST)
        label = dataTransformationResult.get(pc.LABEL)
        featuresColm = dataTransformationResult.get(pc.VECTORFEATURES)
        # featuresColm = "features"

        if trainDataRatio is not None:
            trainData, testData = dataset.randomSplit([trainDataRatio, (1 - trainDataRatio)],
                                                      seed=40)
            ETLOnDatasetStat = {pc.FEATURESCOLM: featuresColm, pc.LABELCOLM: label,
                                pc.TRAINDATA: trainData, pc.TESTDATA: testData,
                                pc.IDNAMEFEATURESORDERED: idNameFeaturesOrdered,
                                pc.DATASET: dataset,
                                pc.INDEXEDFEATURES: indexedFeatures,
                                pc.ONEHOTENCODEDFEATURESLIST: oneHotEncodedFeaturesList}
        else:
            ETLOnDatasetStat = {pc.FEATURESCOLM: featuresColm, pc.LABELCOLM: label,
                                pc.IDNAMEFEATURESORDERED: idNameFeaturesOrdered,
                                pc.DATASET: dataset,
                                pc.INDEXEDFEATURES: indexedFeatures,
                                pc.ONEHOTENCODEDFEATURESLIST: oneHotEncodedFeaturesList}

        return ETLOnDatasetStat

    @staticmethod
    def summaryTable(featuresName, featuresStat):
        statDict = {}
        for name, stat in zip(featuresName.values(),
                              featuresStat.values()):
            statDict[name] = stat
        return statDict

    @staticmethod
    def writeToParquet(fileName, locationAddress, userId, data):
        extention = ".parquet"
        fileName = fileName.upper()
        userId = "" if (userId == "" or userId == None) else userId.upper()
        fileNameWithPath = locationAddress + userId + fileName + extention
        data.write.parquet(fileNameWithPath, mode="overwrite")
        onlyFileName = userId + fileName
        result = {"fileNameWithPath": fileNameWithPath,
                  "onlyFileName": onlyFileName}
        return result

    @staticmethod
    def scaleLocationGraph(label, predictionTargetData, residualsData, modelSheetName, spark):
        sparkContext = spark.sparkContext
        schema = StructType(
            [StructField('stdResiduals', DoubleType(), True), StructField(modelSheetName, DoubleType(), True)])
        try:
            predictionTrainingWithTarget = \
                predictionTargetData.select(label, modelSheetName,
                                            sqrtSpark(absSpark(predictionTargetData[label])).alias("sqrtLabel"))

            predictionTrainingWithTargetIndexing = \
                predictionTrainingWithTarget.withColumn(pc.ROW_INDEX,
                                                        F.monotonically_increasing_id())
            residualsTrainingIndexing = \
                residualsData.withColumn(pc.ROW_INDEX,
                                         F.monotonically_increasing_id())
            residualsPredictiveLabelDataTraining = \
                predictionTrainingWithTargetIndexing.join(residualsTrainingIndexing,
                                                          on=[pc.ROW_INDEX]).sort(
                    pc.ROW_INDEX).drop(pc.ROW_INDEX)
            stdResiduals = \
                residualsPredictiveLabelDataTraining.select("sqrtLabel", modelSheetName,
                                                            (residualsPredictiveLabelDataTraining["residuals"] /
                                                             residualsPredictiveLabelDataTraining[
                                                                 "sqrtLabel"]).alias("stdResiduals"))
            sqrtStdResiduals = \
                stdResiduals.select("stdResiduals", modelSheetName,
                                    sqrtSpark(absSpark(stdResiduals["stdResiduals"])).alias(
                                        "sqrtStdResiduals"))
            sqrtStdResiduals = sqrtStdResiduals.select("stdResiduals", modelSheetName)
            sqrtStdResiduals.na.drop()
            print("scaleLocation plot : success")
        except:
            sqrtStdResiduals = spark.createDataFrame(sparkContext.emptyRDD(), schema)
            print("scaleLocation plot : failed")


        return sqrtStdResiduals

    @staticmethod
    def residualsFittedGraph(residualsData, predictionData, modelSheetName, spark):
        sparkContext = spark.sparkContext
        schema = StructType(
            [StructField(modelSheetName, DoubleType(), True), StructField('residuals', DoubleType(), True)])

        try:
            predictionData = predictionData.select(modelSheetName)
            residualsTrainingIndexing = residualsData.withColumn(pc.ROW_INDEX,
                                                                 F.monotonically_increasing_id())
            predictionTrainingIndexing = predictionData.withColumn(pc.ROW_INDEX,
                                                                   F.monotonically_increasing_id())
            residualsPredictiveDataTraining = \
                predictionTrainingIndexing.join(residualsTrainingIndexing,
                                                on=[pc.ROW_INDEX]).sort(
                    pc.ROW_INDEX).drop(pc.ROW_INDEX)
            residualsPredictiveDataTraining.na.drop()
            print("residual fitted plot : success")
        except:
            residualsPredictiveDataTraining = spark.createDataframe(sparkContext.emptyRDD(), schema)
            print("residual fitted plot : failed")

        return residualsPredictiveDataTraining

    @staticmethod
    def quantileQuantileGraph(residualsData, spark):
        sparkContext = spark.sparkContext
        schema = StructType(
            [StructField('theoryQuantile', DoubleType(), True), StructField('practicalQuantile', DoubleType(), True)])

        try:
            sortedResiduals = residualsData.sort("residuals")
            residualsCount = sortedResiduals.count()
            quantile = []
            for value in range(0, residualsCount):
                quantile.append((value - 0.5) / residualsCount)
            zTheory = []
            for value in quantile:
                zTheory.append(norm.ppf(abs(value)))

            meanStdDev = []
            stat = \
                sortedResiduals.select(meanSpark("residuals"), stddevSpark("residuals"))
            for rows in stat.rdd.toLocalIterator():
                for row in rows:
                    meanStdDev.append(row)
            meanResiduals = meanStdDev[0]
            stdDevResiduals = meanStdDev[1]
            zPractical = []
            for rows in sortedResiduals.rdd.toLocalIterator():
                for row in rows:
                    zPractical.append((row - meanResiduals) / stdDevResiduals)
            quantileTheoryPractical = []
            for theory, practical in zip(zTheory, zPractical):
                quantileTheoryPractical.append([round(theory, 5),
                                                round(practical, 5)])
            '''
            #for future
            schemaQuantile=StructType([StructField("theoryQuantile",DoubleType(),True),
                                       StructField("practicalQuantile",DoubleType(),True)])
            quantileDataframe=spark.createDataFrame(quantileTheoryPractical,schema=schemaQuantile)
            '''
            quantileQuantileData = \
                pd.DataFrame(quantileTheoryPractical, columns=["theoryQuantile",
                                                               "practicalQuantile"])
            quantileQuantileData = spark.createDataFrame(quantileQuantileData)
            quantileQuantileData.na.drop()
            print("Quantile plot : success")
        except:
            quantileQuantileData = spark.createDataFrame(sparkContext.emptyRDD(), schema)
            print("Quantile plot : failed")

        return quantileQuantileData

    # reverting labelIndexing to its original value
    @staticmethod
    def revertIndexToString(dataset, label, indexedLabel):
        from pyspark.sql.functions import regexp_replace
        distinctLabelDataset = dataset.select(label).distinct()
        listOfDistinctValues = list(distinctLabelDataset.select(label).toPandas()[label])
        # for now we are reverting the index column but actully
        # we have to replace the predicted column, so plan accordingly
        if (len(listOfDistinctValues) == len(indexedLabel)):
            for index, val in enumerate(indexedLabel):
                floatIndex = str(float(index))
                dataset = dataset.withColumn(label, regexp_replace(label, floatIndex, val))

    # removing the specialCharacters from the columns name for parquet writing.
    @staticmethod
    def removeSpecialCharacters(columnName):
        colName = re.sub('[^a-zA-Z0-9]', '_', columnName)
        return colName

    # method to create key value pair for statistics
    @staticmethod
    def statsDict(statList, statDict, idNameFeaturesOrdered):
        for index, value in enumerate(statList):
            statDict[index] = round(value, 4)

        return PredictiveUtilities.summaryTable(featuresName=idNameFeaturesOrdered,
                                                featuresStat=statDict)

    @staticmethod
    def performETL(etlInfo):
        #fix this method when u work on prediction.
        datasetAdd = etlInfo.get(pc.DATASETADD)
        relationshipList = etlInfo.get(pc.RELATIONSHIP_LIST)
        relation = etlInfo.get(pc.RELATIONSHIP)
        trainDataRatio = etlInfo.get(pc.TRAINDATALIMIT)
        spark = etlInfo.get(pc.SPARK)
        #delete the prediction column if already existed---
        modelName = etlInfo.get(pc.MODELSHEETNAME)

        #check if the dataset is already passed by the method or not. if not then read the dataset from the obj
        dataset = etlInfo.get(pc.DATASET)
        if(dataset is None):
            dataset = spark.read.parquet(datasetAdd)
            originalDataset = None #setting this none in case of featureAnlaysis and training session
        else:
            originalDataset = dataset
            etlInfo.pop(pc.DATASET)

        if (modelName is not None):
            dataset = dataset.drop(modelName)

        # changing the relationship of the colm
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataset = \
            dataTransformationObj.colmTransformation(
                colmTransformationList=relationshipList) if relation == pc.NON_LINEAR else dataset
        # transformation
        dataTransformationObj = PredictiveDataTransformation(dataset=dataset)
        dataTransformationResult = dataTransformationObj.dataTranform(etlInfo)
        dataset = dataTransformationResult[pc.DATASET]
        categoricalFeatures = dataTransformationResult.get(pc.CATEGORICALFEATURES)
        numericalFeatures = dataTransformationResult.get(pc.NUMERICALFEATURES)
        maxCategories = dataTransformationResult.get(pc.MAXCATEGORIES)
        categoryColmStats = dataTransformationResult.get(pc.CATEGORYCOLMSTATS)
        indexedFeatures = dataTransformationResult.get(pc.INDEXEDFEATURES)
        idNameFeaturesOrdered = dataTransformationResult.get(pc.IDNAMEFEATURESORDERED)
        oneHotEncodedFeaturesList = dataTransformationResult.get(pc.ONEHOTENCODEDFEATURESLIST)
        label = dataTransformationResult.get(pc.LABEL)
        featuresColm = dataTransformationResult.get(pc.VECTORFEATURES)
        isLabelIndexed = dataTransformationResult.get(pc.ISLABELINDEXED)
        # featuresColm = "features"

        if trainDataRatio is not None:
            trainData, testData = dataset.randomSplit([trainDataRatio, (1 - trainDataRatio)],
                                                      seed=40)
            ETLOnDatasetStat = {pc.FEATURESCOLM: featuresColm,
                                pc.LABELCOLM: label,
                                pc.TRAINDATA: trainData,
                                pc.TESTDATA: testData,
                                pc.IDNAMEFEATURESORDERED: idNameFeaturesOrdered,
                                pc.DATASET: dataset,
                                pc.INDEXEDFEATURES: indexedFeatures,
                                pc.ONEHOTENCODEDFEATURESLIST: oneHotEncodedFeaturesList,
                                pc.MAXCATEGORIES: maxCategories,
                                pc.CATEGORYCOLMSTATS: categoryColmStats,
                                pc.CATEGORICALFEATURES: categoricalFeatures,
                                pc.NUMERICALFEATURES: numericalFeatures,
                                pc.ISLABELINDEXED: isLabelIndexed,
                                pc.ORIGINALDATASET:originalDataset}
        else:
            ETLOnDatasetStat = {pc.FEATURESCOLM: featuresColm,
                                pc.LABELCOLM: label,
                                pc.IDNAMEFEATURESORDERED: idNameFeaturesOrdered,
                                pc.DATASET: dataset,
                                pc.INDEXEDFEATURES: indexedFeatures,
                                pc.ONEHOTENCODEDFEATURESLIST: oneHotEncodedFeaturesList,
                                pc.MAXCATEGORIES: maxCategories,
                                pc.CATEGORYCOLMSTATS: categoryColmStats,
                                pc.CATEGORICALFEATURES: categoricalFeatures,
                                pc.NUMERICALFEATURES: numericalFeatures,
                                pc.ISLABELINDEXED: isLabelIndexed,
                                pc.ORIGINALDATASET:originalDataset
                                }

        return ETLOnDatasetStat

    @staticmethod
    def addInternalId(dataset):
        dataset = dataset.drop(pc.DMXINDEX)
        dataset = dataset.withColumn(pc.DMXINDEX, F.monotonically_increasing_id())
        return dataset

    @staticmethod
    def joinDataset(datasetOne, datasetTwo, joinOnColumn):
        dataset = datasetOne.join(datasetTwo, on=[joinOnColumn]).sort(joinOnColumn)
        return dataset



    '''alternative of one hot encoding for sentiment analysis.'''
    @staticmethod
    def countVectorizer(infoData):
        colName = infoData.get(pc.COLMTOENCODE)
        dataset = infoData.get(pc.DATASET)
        encodedColm = infoData.get(pc.ENCODEDCOLM)
        originalColmName = infoData.get(pc.ORIGINALCOLMNAME)
        oneHotEncoderPathMapping = infoData.get(pc.ONEHOTENCODERPATHMAPPING)
        storageLocation = infoData.get(pc.STORAGELOCATION)
        countVectorizer = CountVectorizer(inputCol=colName,
                                          outputCol=encodedColm).fit(dataset)
        '''oneHotEncoderPath = storageLocation + modelId.upper() + PredictiveConstants.ONEHOTENCODED.upper() + PredictiveConstants.PARQUETEXTENSION
        oneHotEncoder.write().overwrite().save(oneHotEncoderPath)
        oneHotEncoderPathMapping.update({
            PredictiveConstants.ONEHOTENCODED: oneHotEncoderPath
        })'''

        oneHotEncoderPath = storageLocation +  pc.ONEHOTENCODED_.upper() + originalColmName.upper() + pc.PARQUETEXTENSION
        countVectorizer.write().overwrite().save(oneHotEncoderPath)
        oneHotEncoderPathMapping.update({
            originalColmName: oneHotEncoderPath
        })

        dataset = countVectorizer.transform(dataset)
        infoData.update({
            pc.ONEHOTENCODERPATHMAPPING: oneHotEncoderPathMapping,
            pc.DATASET: dataset
        })
        return infoData

    @staticmethod
    def featureAssembler(infoData):
        '''requires list of colms to vectorized , dataset and output colm name for feature'''
        colList = infoData.get(pc.COLMTOVECTORIZED)
        if(isinstance(colList, str)):
            colList = [colList]
        dataset = infoData.get(pc.DATASET)
        featuresColm = infoData.get(pc.FEATURESCOLM)
        dataset = dataset.drop(featuresColm)

        featureassembler = VectorAssembler(
            inputCols=colList,
            outputCol=featuresColm, handleInvalid="skip")
        dataset = featureassembler.transform(dataset)
        infoData.update({
            pc.DATASET: dataset
        })
        return infoData

    @staticmethod
    def stringIndexer(infoData):
        colmToIndex = infoData.get(pc.COLMTOINDEX)
        dataset = infoData.get(pc.DATASET)
        indexedColm = infoData.get(pc.INDEXEDCOLM)
        storageLocation = infoData.get(pc.STORAGELOCATION)
        indexerName = colmToIndex + pc.INDEXER
        file = storageLocation + indexerName
        # check if the datatype of the col is integer or float or double. if yes then no need to do the indexing-- sahil.
        '''for now converting each datatypes to string and then indexing it.'''
        dataset = dataset.withColumn(colmToIndex, dataset[colmToIndex].cast(StringType()))
        stringIndexer = StringIndexer(inputCol=colmToIndex, outputCol=indexedColm,
                                     handleInvalid="keep").fit(dataset)
        dataset = stringIndexer.transform(dataset)
        stringIndexer.write().overwrite().save(file)  # will update this later
        indexerPathMapping = infoData.get(pc.INDEXERPATHMAPPING)
        indexerPathMapping.update({colmToIndex: file})
        infoData.update({
            pc.INDEXERPATHMAPPING: indexerPathMapping,
            pc.DATASET: dataset
        })

        return infoData

    @staticmethod
    def indexToString(infoData):
        stringIndexerPath = infoData.get(pc.INDEXERPATH)
        inverterColm = infoData.get(pc.COLMTOINVERT)
        dataset = infoData.get(pc.DATASET)
        stringIndexer = StringIndexerModel.load(stringIndexerPath)
        inverter = IndexToString(inputCol=inverterColm, outputCol=pc.DMXINVERTEDCOLM,
                                 labels=stringIndexer.labels)
        dataset = inverter.transform(dataset)

        #drop the indexed colm and rename the new unindexed colm with the actual one
        dataset = dataset.drop(inverterColm)
        dataset = dataset.withColumnRenamed(pc.DMXINVERTEDCOLM, inverterColm)
        return dataset

    @staticmethod
    def writeToJson(storageLocation, data):
        extension = ".json"
        mode = "w"  # write
        json.dump(data, open(storageLocation + extension, mode))

    @staticmethod
    def duplicateDataset(dataset):
        dataset = PredictiveUtilities.addInternalId(dataset)
        duplicateDataset = dataset
        return dataset, duplicateDataset

    """
    #write to csv
        # finalDataset.drop(pc.DMXSTEMMEDWORDS, pc.DMXSTOPWORDS, pc.DMXTOKENIZED, pc.DMXTAGGEDCOLM,pc.DMXSTEMMEDWORDS, pc.DMXNGRAMS).coalesce(
        #     1).write.mode("overwrite").format("com.databricks.spark.csv").option("header", "true").csv(
        #     "/home/fidel/Documents/knimeStopWordsRohitSirDataWithStem.csv")
    """

    # replace the value of a column based on some condition
    """
    datasetDL = datasetDL.withColumn("original_sentiment_redefined", 
    when(datasetDL["original_sentiment"] == "POS", "positive")
    .when(datasetDL["original_sentiment"] == "NEG", "negative")).show()
    """