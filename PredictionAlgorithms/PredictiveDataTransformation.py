import json
import math

from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, IndexToString
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from pyspark.sql.types import *

from PredictionAlgorithms.PredictiveConstants import PredictiveConstants


class PredictiveDataTransformation():
    def __init__(self, dataset):
        self.dataset = None if dataset == None else dataset

    def nonNumericToString(self, schemaData, dataset):
        categoricalFeatures = []
        numericalFeatures = []

        for schemaVal in schemaData:
            if (str(schemaVal.dataType) == "StringType" or str(schemaVal.dataType) == "TimestampType" \
                    or str(schemaVal.dataType) == "DateType" \
                    or str(schemaVal.dataType) == "BooleanType" \
                    or str(schemaVal.dataType) == "BinaryType"):
                for y in self.featuresColm:
                    if schemaVal.name == y:
                        dataset = dataset.withColumn(y, dataset[y].cast(StringType()))
                        categoricalFeatures.append(schemaVal.name)
            else:
                for y in self.featuresColm:
                    if schemaVal.name == y:
                        numericalFeatures.append(schemaVal.name)

        info = {
            PredictiveConstants.NUMERICALFEATURES: numericalFeatures,
            PredictiveConstants.CATEGORICALFEATURES: categoricalFeatures,
            PredictiveConstants.DATASET: dataset
        }

        return info

    def isLabelIndexed(self, schemaData, label, dataset):
        isLabelIndexed = "no"
        labelIndexedInfo = {}
        labelIndexer = None
        for schemaVal in schemaData:
            if (str(schemaVal.dataType) == "StringType" and schemaVal.name == label):
                labelIndexer = StringIndexer(inputCol=label, outputCol=PredictiveConstants.INDEXED_ + label,
                                             handleInvalid="keep").fit(dataset)
                dataset = labelIndexer.transform(dataset)
                label = PredictiveConstants.INDEXED_ + label
                isLabelIndexed = "yes"
            if (str(schemaVal.dataType) != "StringType" and schemaVal.name == label):
                label = label
                isLabelIndexed = "no"

        labelIndexedInfo.update({
            PredictiveConstants.DATASET: dataset,
            PredictiveConstants.ISLABELINDEXED: isLabelIndexed,
            PredictiveConstants.LABELINDEXER: labelIndexer
        })

        return labelIndexedInfo

    # data transformation
    def dataTranform(self, dataInfo):
        featuresColm = dataInfo.get(PredictiveConstants.FEATURESCOLM)  # featureColmList -replaced
        labelColm = dataInfo.get(PredictiveConstants.LABELCOLM)
        modelSheetName = dataInfo.get(PredictiveConstants.MODELSHEETNAME)
        modelId = dataInfo.get(PredictiveConstants.MODELID)
        storageLocation = dataInfo.get(PredictiveConstants.LOCATIONADDRESS)

        indexerPathMapping = {}
        oneHotEncoderPathMapping = {}

        self.labelColm = None if labelColm == None else labelColm
        self.featuresColm = None if featuresColm == None else featuresColm
        dataset = self.dataset
        vectorizedFeaturescolmName = modelSheetName + PredictiveConstants.DMXFEATURE
        dataset = dataset.drop(vectorizedFeaturescolmName)

        schemaData = dataset.schema

        if self.labelColm is not None:
            for labelName in self.labelColm:
                label = labelName
        else:
            label = self.labelColm

        nonNumericData = self.nonNumericToString(schemaData=schemaData, dataset=dataset)
        categoricalFeatures = nonNumericData.get(PredictiveConstants.CATEGORICALFEATURES)
        numericalFeatures = nonNumericData.get(PredictiveConstants.NUMERICALFEATURES)
        dataset = nonNumericData.get(PredictiveConstants.DATASET)
        schemaData = dataset.schema

        # indexing of label column
        isLabelIndexed = "no"
        if self.labelColm is not None:
            labelIndexedInfo = self.isLabelIndexed(schemaData=schemaData, label=label, dataset=dataset)
            dataset = labelIndexedInfo.get(PredictiveConstants.DATASET)
            isLabelIndexed = labelIndexedInfo.get(PredictiveConstants.ISLABELINDEXED)
            labelIndexer = labelIndexedInfo.get(PredictiveConstants.LABELINDEXER)
            # store the label indexer here.
            if labelIndexer is not None:
                labelIndexerStoragepath = storageLocation + modelId.upper() + label.upper() + PredictiveConstants.INDEXER.upper() + PredictiveConstants.PARQUETEXTENSION
                labelIndexer.save(labelIndexerStoragepath) #correct this
                indexerPathMapping.update({
                    label: labelIndexerStoragepath
                })

        oneHotEncodedFeaturesList = []
        indexedFeatures = []
        nonOneHotEncoded = []
        for colm in categoricalFeatures:
            indexedColmName = PredictiveConstants.INDEXED_ + colm
            oneHotEncodedColName = PredictiveConstants.ONEHOTENCODED_ + colm
            indexer = StringIndexer(inputCol=colm, outputCol=indexedColmName,
                                    handleInvalid="skip").fit(dataset)
            dataset = indexer.transform(dataset)
            '''store the indexer here- saving mechanism should be modelid+colmName+indexer.parquet -->
            not storing the features indexer for now but keeping this for future use.'''
            featuresIndexerPath = storageLocation + modelId.upper() + colm.upper() + PredictiveConstants.INDEXER.upper() + PredictiveConstants.PARQUETEXTENSION
            indexer.write().overwrite().save(featuresIndexerPath)
            indexerPathMapping.update({
                colm: featuresIndexerPath
            })
            rowNo = dataset.select(indexedColmName).distinct().count()
            '''Incase of only one category in the colm or more than one in case of training and prediction
            -- naming it with the onehotencoded colmName to avoid that uncertainity. '''
            if (rowNo == 1):
                nonOneHotEncoded.append(
                    oneHotEncodedColName)  # to avoid the problem when only single value colm is passed at the time of prediction.
            else:
                indexedFeatures.append(indexedColmName)
                oneHotEncodedFeaturesList.append(oneHotEncodedColName)

        oneHotEncoder = OneHotEncoder(inputCols=indexedFeatures,
                                               outputCols=oneHotEncodedFeaturesList,
                                               handleInvalid="error")
        oneHotEncoderPath = storageLocation + modelId.upper() + PredictiveConstants.ONEHOTENCODED.upper() + PredictiveConstants.PARQUETEXTENSION
        oneHotEncoder.write().overwrite().save(oneHotEncoderPath)
        oneHotEncoderPathMapping.update({
            PredictiveConstants.ONEHOTENCODED : oneHotEncoderPath
        })
        oneHotEncoderFit = oneHotEncoder.fit(dataset)
        dataset = oneHotEncoderFit.transform(dataset)

        combinedFeatures = oneHotEncodedFeaturesList + numericalFeatures + nonOneHotEncoded
        categoryColmListDict = {}
        countOfCategoricalColmList = []
        for value in categoricalFeatures:
            listValue = []
            categoryColm = dataset.groupby(value).count()
            countOfCategoricalColmList.append(categoryColm.count())
            categoryColmJson = categoryColm.toJSON()
            for row in categoryColmJson.collect():
                categoryColmSummary = json.loads(row)
                listValue.append(categoryColmSummary)
            categoryColmListDict[value] = listValue

        self.numericalFeatures = numericalFeatures
        self.categoricalFeatures = categoricalFeatures
        if not categoricalFeatures:
            maxCategories = 5
        else:
            maxCategories = max(countOfCategoricalColmList)
        dataset = dataset.drop(vectorizedFeaturescolmName)
        featureassembler = VectorAssembler(
            inputCols=combinedFeatures,
            outputCol=vectorizedFeaturescolmName, handleInvalid="skip")
        dataset = featureassembler.transform(dataset)

        # retrieve the features colm name after onehotencoding
        indexOfFeatures = dataset.schema.names.index(vectorizedFeaturescolmName)
        oneHotEncodedFeaturesDict = dataset.schema.fields[indexOfFeatures].metadata['ml_attr']['attrs']
        idNameFeatures = {}

        if not oneHotEncodedFeaturesDict:
            idNameFeaturesOrderedTemp = None
        else:
            for type, value in oneHotEncodedFeaturesDict.items():
                for subKey in value:
                    idNameFeatures[subKey.get("idx")] = subKey.get("name")
                    idNameFeaturesOrderedTemp = {}
                    for key in sorted(idNameFeatures):
                        idNameFeaturesOrderedTemp[key] = idNameFeatures[key].replace(PredictiveConstants.ONEHOTENCODED_,
                                                                                     "")

        idNameFeaturesOrdered = None if idNameFeaturesOrderedTemp == None else idNameFeaturesOrderedTemp

        # retrieve the label colm name only after label encoding
        indexedLabelNameDict = {}
        if isLabelIndexed == "yes":
            indexOfLabel = dataset.schema.names.index(label)
            indexedLabel = dataset.schema.fields[indexOfLabel].metadata["ml_attr"]["vals"]

            for value in indexedLabel:
                indexedLabelNameDict[indexedLabel.index(value)] = value

        # this code was for vector indexer since it is not stable for now from spark end
        # so will use it in future if needed.
        '''
        vec_indexer = VectorIndexer(inputCol='features', outputCol='vec_indexed_features',
        maxCategories=maxCategories,
                                    handleInvalid="skip").fit(dataset)
        categorical_features = vec_indexer.categoryMaps
        print("Choose %d categorical features: %s" %
              (len(categorical_features), ", ".join(str(k) for k in categorical_features.keys())))
        dataset= vec_indexer.transform(dataset)
        '''

        result = {PredictiveConstants.DATASET: dataset,
                  PredictiveConstants.CATEGORICALFEATURES: categoricalFeatures,
                  PredictiveConstants.NUMERICALFEATURES: numericalFeatures,
                  PredictiveConstants.MAXCATEGORIES: maxCategories,
                  PredictiveConstants.CATEGORYCOLMSTATS: categoryColmListDict,
                  PredictiveConstants.INDEXEDFEATURES: indexedFeatures,
                  PredictiveConstants.LABEL: label,
                  PredictiveConstants.ONEHOTENCODEDFEATURESLIST: oneHotEncodedFeaturesList,
                  PredictiveConstants.INDEXEDLABELNAMEDICT: indexedLabelNameDict,
                  PredictiveConstants.ISLABELINDEXED: isLabelIndexed,
                  PredictiveConstants.VECTORFEATURES: vectorizedFeaturescolmName,
                  PredictiveConstants.IDNAMEFEATURESORDERED: idNameFeaturesOrdered,
                  PredictiveConstants.INDEXERPATHMAPPING: indexerPathMapping,
                  PredictiveConstants.ONEHOTENCODERPATHMAPPING: oneHotEncoderPathMapping
                  }
        return result

    # stats of the each colm
    def dataStatistics(self, categoricalFeatures, numericalFeatures, categoricalColmStat):
        self.categoricalFeatures = None if categoricalFeatures == None else categoricalFeatures
        self.numericalFeatures = None if numericalFeatures == None else numericalFeatures
        categoricalColmStat = None if categoricalColmStat == None else categoricalColmStat
        summaryList = ['mean', 'stddev', 'min', 'max']
        summaryDict = {}
        dataset = self.dataset
        import pyspark.sql.functions as F
        import builtins

        # distint values in colms(categorical only)
        distinctValueDict = {}
        for colm, distVal in categoricalColmStat.items():
            tempList = []
            for value in distVal:
                for key, val in value.items():
                    if key != "count":
                        tempList.append(val)
            distinctValueDict[colm] = len(tempList)

        # stats for numerical colm
        round = getattr(builtins, 'round')
        for colm in self.numericalFeatures:
            summaryListTemp = []
            for value in summaryList:
                summ = list(dataset.select(colm).summary(value).toPandas()[colm])
                summaryListSubTemp = []
                for val in summ:
                    summaryListSubTemp.append(round(float(val), 4))
                summaryListTemp.extend(summaryListSubTemp)
            summaryDict[colm] = summaryListTemp
        summaryList.extend(['skewness', 'kurtosis', 'variance'])
        summaryDict['summaryName'] = summaryList
        summaryDict['categoricalColumn'] = self.categoricalFeatures
        summaryDict["categoricalColmStats"] = distinctValueDict

        skewnessList = []
        kurtosisList = []
        varianceList = []
        skewKurtVarDict = {}
        for colm in self.numericalFeatures:
            skewness = (dataset.select(F.skewness(dataset[colm])).toPandas())
            for i, row in skewness.iterrows():
                for j, column in row.iteritems():
                    skewnessList.append(round(column, 4))
            kurtosis = (dataset.select(F.kurtosis(dataset[colm])).toPandas())
            for i, row in kurtosis.iterrows():
                for j, column in row.iteritems():
                    kurtosisList.append(round(column, 4))
            variance = (dataset.select(F.variance(dataset[colm])).toPandas())
            for i, row in variance.iterrows():
                for j, column in row.iteritems():
                    varianceList.append(round(column, 4))

        for skew, kurt, var, colm in zip(skewnessList, kurtosisList, varianceList, self.numericalFeatures):
            skewKurtVarList = []
            skewKurtVarList.append(skew)
            skewKurtVarList.append(kurt)
            skewKurtVarList.append(var)
            skewKurtVarDict[colm] = skewKurtVarList

        for (keyOne, valueOne), (keyTwo, valueTwo) in zip(summaryDict.items(), skewKurtVarDict.items()):
            if keyOne == keyTwo:
                valueOne.extend(valueTwo)
                summaryDict[keyOne] = valueOne
        return summaryDict

    def colmTransformation(self, colmTransformationList):
        colmTransformationList = None if colmTransformationList == None else colmTransformationList
        dataset = self.dataset

        def relationshipTransform(dataset, colmTransformationList):
            # creating the udf
            def log_list(x):
                try:
                    return math.log(x, 10)
                except Exception as e:
                    print('(log error)number should not be less than or equal to zero: ' + str(e))
                    pass
                # finally:
                #     print('pass')
                #     pass

            def exponent_list(x):
                try:
                    return math.exp(x)
                except Exception as e:
                    print('(exception error)number should not be large enough to get it infinity: ' + str(e))

            def square_list(x):
                try:
                    return x ** 2
                except Exception as e:
                    print('(square error)number should not be negative: ' + str(e))

            def cubic_list(x):
                try:
                    return x ** 3
                except Exception as e:
                    print('(cubic error)number should not be negative: ' + str(e))

            def quadritic_list(x):
                try:
                    return x ** 4
                except Exception as e:
                    print('(quadratic error )number should not be negative: ' + str(e))

            def sqrt_list(x):
                try:
                    return math.sqrt(x)
                except Exception as e:
                    print('(sqare root error) number should not be negative: ' + str(e))

            square_list_udf = udf(lambda y: square_list(y), FloatType())
            log_list_udf = udf(lambda y: log_list(y), FloatType())
            exponent_list_udf = udf(lambda y: exponent_list(y), FloatType())
            cubic_list_udf = udf(lambda y: cubic_list(y), FloatType())
            quadratic_list_udf = udf(lambda y: quadritic_list(y), FloatType())
            sqrt_list_udf = udf(lambda y: sqrt_list(y), FloatType())

            # spark.udf.register("squaredWithPython", square_list)
            # square_list_udf = udf(lambda y: square_list(y), ArrayType(FloatType))
            # square_list_udf = udf(lambda y: exponent_list(y), FloatType())
            # # dataset.select('MPG', square_list_udf(col('MPG').cast(FloatType())).alias('MPG')).show()
            # dataset.withColumn('MPG', square_list_udf(col('MPG').cast(FloatType()))).show()
            # Relationship_val = 'square_list'
            # Relationship_colm = ["CYLINDERS", "WEIGHT", "ACCELERATION", "DISPLACEMENT"]
            # Relationship_model = ['log_list', 'exponent_list', 'square_list', 'cubic_list', 'quadritic_list',
            #                       'sqrt_list']

            for key, value in colmTransformationList.items():
                if key == 'square_list':
                    if len(value) == 0:
                        print('length is null')
                    else:
                        for colm in value:
                            # Relationship_val.strip("'")
                            # square_list_udf = udf(lambda y: square_list(y), FloatType())
                            dataset = dataset.withColumn(colm, square_list_udf(col(colm).cast(FloatType())))
                if key == 'log_list':
                    if len(value) == 0:
                        print('length is null')
                    else:
                        for colm in value:
                            # Relationship_val.strip("'")
                            # square_list_udf = udf(lambda y: square_list(y), FloatType())
                            dataset = dataset.withColumn(colm, log_list_udf(col(colm).cast(FloatType())))
                if key == 'exponent_list':
                    if len(value) == 0:
                        print('length is null')
                    else:
                        for colm in value:
                            # Relationship_val.strip("'")
                            # square_list_udf = udf(lambda y: square_list(y), FloatType())
                            dataset = dataset.withColumn(colm, exponent_list_udf(col(colm).cast(FloatType())))
                if key == 'cubic_list':
                    if len(value) == 0:
                        print('length is null')
                    else:
                        for colm in value:
                            # Relationship_val.strip("'")
                            # square_list_udf = udf(lambda y: square_list(y), FloatType())
                            dataset = dataset.withColumn(colm, cubic_list_udf(col(colm).cast(FloatType())))
                if key == 'quadratic_list':
                    if len(value) == 0:
                        print('length is null')
                    else:
                        for colm in value:
                            # Relationship_val.strip("'")
                            # square_list_udf = udf(lambda y: square_list(y), FloatType())
                            dataset = dataset.withColumn(colm, quadratic_list_udf(col(colm).cast(FloatType())))
                if key == 'sqrt_list':
                    if len(value) == 0:
                        print('length is null')
                    else:
                        for colm in value:
                            # Relationship_val.strip("'")
                            # square_list_udf = udf(lambda y: square_list(y), FloaType())
                            dataset = dataset.withColumn(colm, sqrt_list_udf(col(colm).cast(FloatType())))
                else:
                    print('not found')

            return (dataset)

        # result={"colmTransformationList":colmTransformationList, "dataset":dataset}
        result = relationshipTransform(dataset, colmTransformationList)
        return result
