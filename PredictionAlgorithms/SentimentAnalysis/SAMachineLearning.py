from abc import ABC, abstractmethod
from PredictionAlgorithms.SentimentAnalysis.SentimentAnalysis import SentimentAnalysis
from PredictionAlgorithms.SentimentAnalysis.TextProcessing import TextProcessing
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu

class SAMachineLearning(SentimentAnalysis):

    def sentimentAnalysis(self, sentimentInfoData):

        spark = sentimentInfoData.get(pc.SPARK)
        datasetPath = sentimentInfoData.get(pc.SENTIMENTDATASETPATH)
        dataset = spark.read.parquet(datasetPath)
        dataset = pu.addInternalId(dataset)
        sentimentInfoData.update({pc.DATASET: dataset})

        isNgram = sentimentInfoData.get(pc.ISNGRAM)
        sentimentDataset = self.textPreProcessing(sentimentInfoData)  # do the oneHot Encoding after that.
        textProcessing = TextProcessing(sparkSession=spark)
        sentimentDataset = textProcessing.lemmatization(sentimentDataset, pc.DMXSTOPWORDS)
        # sentimentDataset = textProcessing.sparkLemmatizer(sentimentDataset, pc.DMXSTOPWORDS)
        if(isNgram):
            ngramPara = sentimentInfoData.get(pc.NGRAMPARA)
            sentimentDataset = textProcessing.ngrams(sentimentDataset, pc.DMXLEMMATIZED, ngramPara)  # with n-grams

        modelName = sentimentInfoData.get(pc.MODELSHEETNAME)
        labelColm = sentimentInfoData.get(pc.LABELCOLM)
        indexedColm = pc.INDEXED_ + labelColm
        encodedColm = pc.ONEHOTENCODED_ + pc.DMXLEMMATIZED
        featuresColm = modelName + pc.DMXFEATURE

        sentimentInfoData.update({pc.COLMTOENCODE: pc.DMXLEMMATIZED,
                                  pc.DATASET: sentimentDataset,
                                  pc.COLMTOINDEX: labelColm,
                                  pc.INDEXEDCOLM: indexedColm,
                                  pc.ENCODEDCOLM: encodedColm,
                                  pc.COLMTOVECTORIZED: encodedColm,
                                  pc.FEATURESCOLM: featuresColm,
                                  pc.ORIGINALCOLMNAME: labelColm
                                  })

        sentimentInfoData = pu.stringIndexer(sentimentInfoData)  # after this will get the indexed label
        # at the place of countvectorizer we can use the tf-idf method of spark too...
        sentimentInfoData = pu.countVectorizer(sentimentInfoData)  # using the lemmatized colm for now.
        if(isNgram):
            sentimentInfoData.update({
                pc.COLMTOENCODE: pc.DMXNGRAMS,
                pc.ENCODEDCOLM: pc.ONEHOTENCODED_ + pc.DMXNGRAMS
            })
            sentimentInfoData = pu.countVectorizer(sentimentInfoData)
            sentimentInfoData.update({pc.COLMTOVECTORIZED: [pc.ONEHOTENCODED_ + pc.DMXLEMMATIZED, pc.ONEHOTENCODED_ + pc.DMXNGRAMS]})

        sentimentInfoData = pu.featureAssembler(sentimentInfoData)  # creating feature vector

        return sentimentInfoData

    @abstractmethod
    def evaluation(self, infoData):
        raise NotImplementedError("subClass must implement abstract method")


    def invertIndexColm(self, infoData):
        originalColm = infoData.get(pc.COLMTOINDEX)
        stringIndexerPath = (infoData.get(pc.INDEXERPATHMAPPING)).get(originalColm)
        inverterColm = infoData.get(pc.PREDICTIONCOLM)
        testDataset = infoData.get(pc.TESTDATA)
        trainDataset = infoData.get(pc.TRAINDATA)
        infoData.update({
            pc.INDEXERPATH: stringIndexerPath,
            pc.COLMTOINVERT: inverterColm
        })
        """
        run the indexing part on test and train dataset seperately since needs to show the user accordingly
        """
        infoData.update({pc.DATASET: trainDataset})
        trainDataset = pu.indexToString(infoData)
        infoData.update({pc.DATASET: testDataset})
        testDataset = pu.indexToString(infoData)

        infoData.update({
            pc.TRAINDATA: trainDataset,
            pc.TESTDATA: testDataset
        })

        return infoData