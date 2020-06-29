from pyspark import AccumulatorParam
from pyspark.sql import SparkSession
import re

sparkTest = SparkSession.builder.appName('testCASe').master('local[*]').getOrCreate()
sc = sparkTest.sparkContext

class DictParam(AccumulatorParam):
    def zero(self,  value = ""):
        return dict()

    def addInPlace(self, acc1, acc2):
        acc1.update(acc2)


if  __name__== "__main__":

    df = sparkTest.read.csv("/home/fidel/Documents/sentimentMovieReviewDataset.csv")
    df_rdd = df.rdd

    dict1 = sc.accumulator({}, DictParam())


    def file_read(line):
        global dict1
        # ls = re.split(',', line)
        # dict1+={ls[0]:ls[1]}
        print(line)
        return line


    rdd = df_rdd.mapPartitionsWithIndex(file_read, True)
    print(dict1.count())
    sparkTest.stop()