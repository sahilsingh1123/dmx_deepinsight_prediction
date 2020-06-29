from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.stat import Correlation


class PredictiveStatisticalTest():
    def __init__(self, dataset, features, labelColm):
        self.dataset = dataset
        self.features = features
        self.labelColm = labelColm

    def pearsonTest(self):
        dataset = self.dataset
        labelColm = self.labelColm
        features = self.features
        labelColm = [labelColm]

        allColms = labelColm + features

        featureAssembler = VectorAssembler(
            inputCols=allColms, outputCol="allColmsVectorized", handleInvalid="skip")
        allColmsVectorizedDataset = featureAssembler.transform(dataset)
        r1p = Correlation.corr(allColmsVectorizedDataset, "allColmsVectorized").head()
        print("pearson correlation matrix : \n : " + str(r1p[0]))
        pearson_matrix = r1p[0].toArray().tolist()
        pearsonMatrix = []
        for everylist in pearson_matrix:
            insideList = []
            for listinlist in everylist:
                insideList.append(round(listinlist, 4))
            pearsonMatrix.append(insideList)
        pearson_value_d = []
        for x in r1p[0].toArray():
            pearson_value_d.append(round(x[0], 4))
            # pearson_value_d.append(x[0])
        pearson_value = {}
        for col, val in zip(allColms, pearson_value_d):
            pearson_value[col] = val
        #
        # r1s = Correlation.corr(allColmsVectorizedDataset, "allColmsVectorized", "spearman").head()
        result_pearson = {'pearson_value': pearson_value,
                          'matrix': pearsonMatrix}
        return result_pearson

    def chiSquareTest(self, categoricalFeatures, maxCategories):
        dataset = self.dataset
        labelColm = self.labelColm
        features = self.features
        length = features.__len__()

        featureassembler = VectorAssembler(
            inputCols=self.features,
            outputCol="featuresChiSquare", handleInvalid="skip")
        dataset = featureassembler.transform(dataset)

        vec_indexer = VectorIndexer(inputCol="featuresChiSquare",
                                    outputCol='vecIndexedFeaturesChiSqaure',
                                    maxCategories=maxCategories,
                                    handleInvalid="skip").fit(dataset)

        categorical_features = vec_indexer.categoryMaps
        print("Chose %d categorical features: %s" %
              (len(categorical_features), ", ".join(str(k) for k in categorical_features.keys())))
        dataset = vec_indexer.transform(dataset)
        # using chi selector
        selector = ChiSqSelector(numTopFeatures=length, featuresCol="vecIndexedFeaturesChiSqaure",
                                 outputCol="selectedFeatures",
                                 labelCol=labelColm)
        result = selector.fit(dataset).transform(dataset)
        # runnin gfor the chi vallue test

        r = ChiSquareTest.test(result, "selectedFeatures", labelColm).head()
        p_values = list(r.pValues)
        PValues = []
        for val in p_values:
            PValues.append(round(val, 4))
        print(PValues)
        dof = list(r.degreesOfFreedom)
        stats = list(r.statistics)
        statistics = []
        for val in stats:
            statistics.append(round(val, 4))
        chiSquareDict = {}
        for pval, doF, stat, colm in zip(PValues, dof, statistics, categoricalFeatures):
            chiSquareDict[colm] = pval, doF, stat
        chiSquareDict['summaryName'] = ['pValue', 'DoF', 'statistics']
        result = {'pvalues': chiSquareDict}

        return result
