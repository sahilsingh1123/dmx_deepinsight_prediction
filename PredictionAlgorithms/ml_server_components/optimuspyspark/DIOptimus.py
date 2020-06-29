from optimus import Optimus
from pyspark.sql import SparkSession
from termcolor import colored


class DIOptimus(type):

    @classmethod
    def __prepare__(metacls, name, bases, **kargs):
        # kargs = {"myArg1": 1, "myArg2": 2}
        return super().__prepare__(name, bases, **kargs)

    def __new__(metacls, name, bases, namespace, **kargs):
        # kargs = {"myArg1": 1, "myArg2": 2}
        return super().__new__(metacls, name, bases, namespace)

        # DO NOT send "**kargs" to "type.__new__".  It won't catch them and
        # you'll get a "TypeError: type() takes 1 or 3 arguments" exception.

    def __init__(cls, name, bases, attrs, sparkURL='',**kwargs):
        print("inside DIOptimus init function", cls, name, bases, attrs, **kwargs)
        super(DIOptimus, cls).__init__(name, bases, attrs)
        cls.instance = None

    def __call__(cls, *args, sparkURL=None,**kwargs):
        print("Inside the call function", cls, *args, **kwargs)
        if cls.instance is None:
            cls.instance = super().__call__(*args, **kwargs)
            cls.initOptimus(sparkURL)
        return cls.instance

    def initOptimus(cls,  sparkURL=None):
        print(colored("Optimus Should Be initalized only once", 'blue'))
        cls.optimus = Optimus()
        if sparkURL is not None:
            cls.optimus.stop()
            spark = SparkSession.builder.master(sparkURL).appName("DMXDeepInsightpy").getOrCreate()
            cls.optimus = Optimus(spark, verbose=True)

    pass


class Manager(metaclass=DIOptimus, sparkURL=None):
    meta_args = ['sparkURL']

    def getOptimus(cls):
        print("************ Get Optimus Instance ************")
        return cls.optimus


def getmanager(sparkURL=None):
    M = Manager(sparkURL=sparkURL)
    print(M.instance)

    # for t in int,float,dict,list,tuple:
    #     print(type(t))
    #
    # print(type(type))
    #
    #
    # print(M.__class__)
    # print(type(M))
    # print(M.__class__ is type(M))
    #
    #
    #
    # class Foo:
    #     pass
    #
    # x = Foo()
    # print(x)
    #
    # Bar = type('Bar',(Foo,),dict(attr=100))
    # x = Bar()
    # print(x.attr)
    #
    # print(x.__class__)
    # print(x.__class__.__bases__)

    return M
