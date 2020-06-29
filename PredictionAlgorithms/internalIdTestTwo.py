from pyspark import AccumulatorParam
from pyspark.sql import SparkSession

sparkTest = SparkSession.builder.appName('testCASe').master('local[*]').getOrCreate()
sc = sparkTest.sparkContext

class DictParam(AccumulatorParam):
    def zero(self,  value = ""):
        return dict()

    def addInPlace(self, acc1, acc2):
        acc1.update(acc2)

def count_in_a_partition(idx, iterator):
  global dict1
  count = 0
  for _ in iterator:
    count += 1
  print("\n" , count,idx)
  dict1 +={idx:count}
  return idx, count


if (__name__=='__main__'):
    dict1 = sc.accumulator({}, DictParam())

    data = sc.parallelize([
        1, 2, 3, 4
    ], 4)

    data.mapPartitionsWithIndex(count_in_a_partition).collect()
    print(dict1)