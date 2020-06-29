from pyspark.sql.types import *
from pyspark.sql.functions import regexp_replace, Column, col, udf, array_contains, posexplode_outer, lit, \
    monotonically_increasing_id, array_join, concat_ws
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, NGram
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc
from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities as pu
from PredictionAlgorithms.SentimentAnalysis.StopWords import StopWords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sparknlp.annotator import Lemmatizer, LemmatizerModel
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.annotator import *
from sparknlp.base import *
import pandas as pd


class TextProcessing():
    spark: None

    def __init__(self, sparkSession):
        global spark
        spark = sparkSession

    """
    :return dataset with no null value
    """
    def toStringDatatype(self, dataset, colName):
        dataset.na.drop()
        datasetSchema = dataset.schema

        for schemaVal in datasetSchema:
            if (str(schemaVal.dataType) == "TimestampType"
                    or str(schemaVal.dataType) == "DateType"
                    or str(schemaVal.dataType) == "BooleanType"
                    or str(schemaVal.dataType) == "BinaryType"
                    or str(schemaVal.dataType) == "DoubleType"
                    or str(schemaVal.dataType) == "IntegerType"):
                if (schemaVal.name == colName):
                    dataset = dataset.withColumn(colName, dataset[colName].cast(StringType()))

        return dataset

    """replaces all the special character from the string"""

    def replaceSpecialChar(self, dataset, colName):
        regexIgnore = "[^a-zA-Z ]"
        regexInclude = "[,.]"
        dataset = dataset.withColumn(colName, regexp_replace(col(colName), regexIgnore, ""))
        dataset = dataset.withColumn(colName, regexp_replace(col(colName), regexInclude, " "))

        # dataset = self.createToken(dataset, colName)
        return dataset

    """this method create token for each word and also convert the case to lower"""

    def createToken(self, dataset, colName):
        dataset = dataset.drop(pc.DMXTOKENIZED)
        sentimentTokenizer = RegexTokenizer(inputCol=colName, outputCol=pc.DMXTOKENIZED,
                                            toLowercase=True,
                                            pattern="\\W")
        dataset = sentimentTokenizer.transform(dataset)
        return dataset

    '''
    :desc - removes all the stopwords.
    :params - spark-dataframe, column- array of Strings
    :return- spark-dataframe
    '''
    def stopWordsRemover(self, dataset, colName):
        dataset = dataset.drop(pc.DMXSTOPWORDS)
        stopWordsList = StopWords.stopWordsKNIME
        sentimentStopWordRemover = StopWordsRemover(inputCol=colName,
                                                    outputCol=pc.DMXSTOPWORDS)  # .loadDefaultStopWords("english") #change the language based on user preference.
        dataset = sentimentStopWordRemover.transform(dataset)
        return dataset

    def stemming(self, dataset, colName):
        dataset = dataset.drop(pc.DMXSTEMMEDWORDS)
        stemmer = PorterStemmer(mode="NLTK_EXTENSIONS")
        stemmerUDF = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
        stemmedDf = dataset.withColumn(pc.DMXSTEMMEDWORDS, stemmerUDF(colName))
        return stemmedDf

    def lemmatization(self, dataset, colName):
        dataset = dataset.drop(pc.DMXLEMMATIZED)
        lemmatizer = WordNetLemmatizer()
        lemmatizerUDF = udf(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens], ArrayType(StringType()))
        dataset = dataset.withColumn(pc.DMXLEMMATIZED, lemmatizerUDF(colName))
        return dataset

    @staticmethod
    def ngrams(dataset, colName, number):
        dataset = dataset.drop(pc.DMXNGRAMS)
        ngram = NGram(n=number, inputCol=colName, outputCol=pc.DMXNGRAMS)
        dataset = ngram.transform(dataset)
        return dataset

    def sparkLemmatizer(self,dataset, colName, lemmatizedModelPath):
        dataset = dataset.select(pc.DMXINDEX, concat_ws(",", dataset[colName]).alias(colName))
        dataset = dataset.withColumn(colName, regexp_replace(col(colName), ",", " "))
        dataset = dataset.drop('lemma','document')

        documentAssembler = DocumentAssembler()\
            .setInputCol(colName)\
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        lemmaModel = LemmatizerModel.load(lemmatizedModelPath)\
            .setInputCols(["document", "token"]).setOutputCol("lemma")

        #this needs to be added for pos tagging in text analytics.
        posModel = PerceptronModel.load("/dev/dmxdeepinsight/models/POSTaggerSparkNLP").setInputCols(
            ["document", "lemma"]).setOutputCol("posTagged")

        finisher = Finisher() \
            .setInputCols(["lemma"]) \
            .setOutputCols(["ntokens"]) \
            .setOutputAsArray(True) \
            .setCleanAnnotations(True)

        lemmatizerPipeline = Pipeline(
            stages=[documentAssembler, tokenizer, lemmaModel])
        dataset = lemmatizerPipeline.fit(dataset).transform(dataset)
        dataset =  dataset.withColumn(pc.DMXLEMMATIZED, dataset["lemma.result"])

        indexList = []
        lemmatizedList = []
        for index, row in enumerate(dataset.select(pc.DMXINDEX, pc.DMXLEMMATIZED).rdd.toLocalIterator()):
            dmxIndex = row[0]
            rowList = row[1]
            rowListPoped = rowList.pop(0)
            indexList.append(dmxIndex)
            lemmatizedList.append(rowList)
        dataset = self.getExtractedLemmaData(dataset, indexList, lemmatizedList)

        return dataset

    def getExtractedLemmaData(self, dataset, indexList, lemmatizedList):
        zipData = zip(indexList, lemmatizedList)
        columnList = [pc.DMXINDEX, pc.DMXLEMMATIZED]
        pandasDataframe = pd.DataFrame(zipData, columns=columnList)
        lemmatizedDataset = spark.createDataFrame(pandasDataframe)
        dataset = dataset.drop(pc.DMXLEMMATIZED)
        dataset = pu.joinDataset(dataset, lemmatizedDataset, pc.DMXINDEX)
        return dataset

    # frequency of each words in the document per row.
    """
    from pyspark.sql.functions import split, explode, col
    datasetTest = dataset.withColumn("explode", explode(col(pc.DMXSTOPWORDS)))
    datasetTest.groupBy("explode",pc.DMXINDEX).count().sort(pc.DMXINDEX, ascending=True).show(35)
    datasetTest.groupBy("explode").agg(countDistinct(pc.DMXINDEX)) --> count occurance in no of rows in doc.
    """

    # pretrained model for sentiment analysis- just pass the text column and it will return
    # the result with positive and negative sentiment of each row.
    """
    spark--nlp
    --> joining sparkSession of spark-nlp with pyspark
    import sys
    import time
    import sparknlp
    
    from pyspark.sql import SparkSession
    packages = [
        'JohnSnowLabs:spark-nlp: 2.4.2'
    ]
    spark = SparkSession \
        .builder \
        .appName("ML SQL session") \
        .config('spark.jars.packages', ','.join(packages)) \
        .config('spark.executor.instances','2') \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory","2g") \
        .getOrCreate()
    
    
    --> pretrained pipeline
    from sparknlp.pretrained import PretrainedPipeline
    pipelineSentiment = PretrainedPipeline('analyze_sentiment', 'en')
    datasetTest = datasetTest.withColumnRenamed("Text", "text")
    datasetTest = datasetTest.withColumnRenamed("sentiment", "original_sentiment")
    sentimentPrediction = pipelineSentiment.transform(datasetTest)
    
    --> viveknSentiment analysis
    from sparknlp.pretrained import ViveknSentimentModel, ViveknSentimentApproach
    vivekSentimentApproach = ViveknSentimentApproach().setSentimentCol("original_sentiment").setInputCols(["sentence", "stopWordsRemoved"]).setOutputCol("viveknSentiment")
    vivekDataset = vivekSentimentApproach.fit(cleaned)
    
    --> dataTransformation-
    from sparknlp.pretrained import PretrainedPipeline
    explainDL = PretrainedPipeline("explain_document_dl", "en")
    datasetDL = explainDL.transform(datasetDL)
    from sparknlp.annotator import StopWordsCleaner
    stopWordsRemover = StopWordsCleaner().setInputCols(["lemma"]).setOutputCol("stopWordsRemoved")
    datasetDL = stopWordsRemover.transform(datasetDL)
    
    --> finisher-- getting original column from spark-nlp operation
    from sparknlp.base import *
    finisher = Finisher() \
      .setInputCols(["stopWordsRemoved"]) \
      .setOutputCols(["finishedArrayOfString"]) \
      .setIncludeMetadata(False)
    finishedDataset = finisher.transform(cleaned)
    
    
    
    -------------------------------------------------------->
    --> all scratch to end of the system
    
    from sparknlp.annotator import *
    from sparknlp.common import *
    from sparknlp.base import *
    
    from pyspark.ml import Pipeline
    
    
    document_assembler = DocumentAssembler() \
        .setInputCol("comment") \
        .setOutputCol("document")
        
    sentence_detector = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence") \
        .setUseAbbreviations(True)
        
    tokenizer = Tokenizer() \
      .setInputCols(["sentence"]) \
      .setOutputCol("token")
    
    stemmer = Stemmer() \
        .setInputCols(["token"]) \
        .setOutputCol("stem")
        
    normalizer = Normalizer() \
        .setInputCols(["stem"]) \
        .setOutputCol("normalized")
    
    finisher = Finisher() \
        .setInputCols(["normalized"]) \
        .setOutputCols(["ntokens"]) \
        .setOutputAsArray(True) \
        .setCleanAnnotations(True)
        
        --> stopWordsRemover can be used before finisher this.
    
    nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, stemmer, normalizer, finisher])
    --> after this fit and transform it with the dataset.
    
    -- coming to spark end
    from pyspark.ml import feature as spark_ft

    stopWords = spark_ft.StopWordsRemover.loadDefaultStopWords('english')
    sw_remover = spark_ft.StopWordsRemover(inputCol='ntokens', outputCol='clean_tokens', stopWords=stopWords)
    tf = spark_ft.CountVectorizer(vocabSize=500, inputCol='clean_tokens', outputCol='tf')
    idf = spark_ft.IDF(minDocFreq=5, inputCol='tf', outputCol='idf')
    
    feature_pipeline = Pipeline(stages=[sw_remover, tf, idf])
    feature_model = feature_pipeline.fit(train)
    
    --> after training any model for evaluation we will be using this approach
    pred_df = preds.select('comment', 'label', 'prediction').toPandas()
    
    import pandas as pd
    from sklearn import metrics as skmetrics
    pd.DataFrame(
        data=skmetrics.confusion_matrix(pred_df['label'], pred_df['prediction']),
        columns=['pred ' + l for l in ['0','1']],
        index=['true ' + l for l in ['0','1']]
    )
    print(skmetrics.classification_report(pred_df['label'], pred_df['prediction'], 
    
    -- stop spark session here.
    
    
    """


