import json

import pyspark
from flask import Flask
from flask import Response
from flask import jsonify
from flask import request
from pyspark.sql import SparkSession

from PredictionAlgorithms.CreateRequestData import CreateRequestData
from PredictionAlgorithms.PredictiveConstants import PredictiveConstants
from PredictionAlgorithms.PredictiveExceptionHandling import PredictiveExceptionHandling
from PredictionAlgorithms.PredictiveFeatureAnalysis.PredictiveFeatureAnalysis import PredictiveFeatureAnalysis
from PredictionAlgorithms.PredictivePerformPrediction import PredictivePerformPrediction
from PredictionAlgorithms.PredictiveRegression.PredictiveGradientBoostRegression import \
    PredictiveGradientBoostRegression
from PredictionAlgorithms.PredictiveRegression.PredictiveLinearRegression import PredictiveLinearRegression
from PredictionAlgorithms.PredictiveRegression.PredictiveRandomForestRegression import PredictiveRandomForestRegression
from PredictionAlgorithms.PredictiveRegression.PredictiveRidgeRegression import PredictiveRidgeRegression
from PredictionAlgorithms.PredictiveRegression.PredictiveLassoRegression import PredictiveLassoRegression
from PredictionAlgorithms.ml_server_components import FPGrowth
from PredictionAlgorithms.ml_server_components import Forecasting
from PredictionAlgorithms.ml_server_components import KMeans
from PredictionAlgorithms.ml_server_components import SentimentAnalysis
from PredictionAlgorithms.SentimentAnalysis.ViveknPretrainedModel import ViveknPretrainedModel
from PredictionAlgorithms.SentimentAnalysis.LexiconSA import LexiconSA
from pyspark import SparkContext

# spark = \
#     SparkSession.builder.appName('DMXPredictiveAnalytics')\
#         .config("spark.jars", "/home/fidel/cache_pretrained/sparknlpFATjar.jar")\
#         .master('local[*]').getOrCreate() #sahil- fix the sparknlpJar location
# spark.sparkContext.setLogLevel('ERROR')

spark = \
    SparkSession.builder.appName('DMXPredictiveAnalytics')\
        .config("spark.jars", "/home/fidel/cache_pretrained/sparknlpFATjar.jar")\
        .master('local[*]').getOrCreate() #sahil- fix the sparknlpJar location
spark.sparkContext.setLogLevel('ERROR')

'''
pyinstaller --clean DIOptimus.spec
cp  -r pkgs/* dist/DIOptimus/
cd dist

'''

# conf = pyspark.SparkConf().setAppName("predictive_Analysis").setMaster("spark://fidel-Latitude-E5570:7077") #need to get the master url from the java end.
# sc = SparkContext(conf=conf)
# spark = SparkSession(sc)
# spark.read.parquet("hdfs://sahil:9000/dev/dmxdeepinsight/datasets/053E5497-0595-4912-B470-C2A1B0A78F58TESTPREDICTEDVSACTUAL.parquet").show()

app = Flask(__name__)


@app.route("/prediction", methods=["POST", "GET"])
def root():
    response = Response(content_type="application/json")
    responseData = {}
    requestString = request.data.decode("utf-8")
    requestData = json.loads(requestString)
    predictiveData = {}
    requestType = requestData.get(PredictiveConstants.REQUESTTYPE)
    subRequestType = requestData.get(PredictiveConstants.SUBREQUESTTYPE)
    print(subRequestType)

    try:
        createRequestData = CreateRequestData()
        if ("PredictiveAlgorithm".__eq__(requestType)):
            predictiveData = createRequestData.createRegressionModelData(requestData)
            predictiveData[PredictiveConstants.SPARK] = spark

            if (PredictiveConstants.LINEAR_REG.__eq__(subRequestType)):
                predictiveLinearRegression = PredictiveLinearRegression()
                responseData = predictiveLinearRegression.linearRegression(predictiveData)
            elif (PredictiveConstants.RIDGE_REG.__eq__(subRequestType)):
                predictiveRidgeRegression = PredictiveRidgeRegression()
                responseData = predictiveRidgeRegression.ridgeRegression(predictiveData)
            elif (PredictiveConstants.LASSO_REG.__eq__(subRequestType)):
                predictiveLassoRegression = PredictiveLassoRegression()
                responseData = predictiveLassoRegression.lassoRegression(predictiveData)
            elif (PredictiveConstants.RANDOMFORESTALGO.__eq__(subRequestType)):
                predictiveRandomForestRegression = PredictiveRandomForestRegression()
                responseData = predictiveRandomForestRegression.randomForestRegression(predictiveData)
            elif (PredictiveConstants.GRADIENTBOOSTALGO.__eq__(subRequestType)):
                predictiveGradientBoostRegression = PredictiveGradientBoostRegression()
                responseData = predictiveGradientBoostRegression.gradientBoostRegression(predictiveData)

        elif ("FeatureSelection".__eq__(requestType)):
            predictiveData = createRequestData.createFeatureSelectionData(requestData=requestData)
            predictiveData[PredictiveConstants.SPARK] = spark

            predictiveFeatureAnalysis = PredictiveFeatureAnalysis()
            responseData = predictiveFeatureAnalysis.featureSelection(predictiveData)

        elif ("prediction".__eq__(requestType)):
            predictiveData = createRequestData.createPredictionData(requestData)
            predictiveData[PredictiveConstants.SPARK] = spark

            predictivePerformPrediction = PredictivePerformPrediction()
            responseData = predictivePerformPrediction.prediction(predictiveData)

        responseData["run_status"] = "success"
        print(responseData.get("run_status"))

    except Exception as e:
        print('exception is = ' + str(e))
        responseData = PredictiveExceptionHandling.exceptionHandling(e)
    return jsonify(success='success', message='it was a success', data=responseData)


@app.route("/forecasting", methods=["POST", "GET"])
def forecasting():
    response = Response(content_type="application/json")
    requestString = request.data.decode("utf-8")
    requestData = json.loads(requestString)
    print("Request data ", requestData)

    j = json.loads(requestString)
    algorithm = j['algorithm']
    data = j['data']
    response_data = ''
    print(algorithm)
    try:
        if algorithm == 'kmeans':
            response_data = KMeans.perform_k_means(data=data, no_of_clusters=j['number_of_clusters'])
        elif algorithm == 'fp-growth':
            print('This is a FP-Growth request!!!!')
            response_data = FPGrowth.perform_fp_growth(data=data)
            print('Sending FP-Growth Response!!!')
        elif algorithm == 'sentimentAnalysis':
            response_data = SentimentAnalysis.perform_sentiment_analysis(data=data)
        elif algorithm == 'forecasting':
            alpha = j.get("alpha")
            beta = j.get("beta")
            gamma = j.get("gamma")
            isTrending = j.get("isTrend")
            isSeasonal = j.get("isSeason")
            seasonalPeriodsManual = j.get("seasonality")
            seasonalP = j.get("seasonalP")
            seasonalD = j.get("seasonalD")
            seasonalQ = j.get("seasonalQ")
            data = data
            count = j.get('count')
            len_type = j.get('len_type')
            model_type = j.get('model_type')
            trendType = j.get('trendType')
            seasonType = j.get('seasonType')
            forecastAlgorithm = j.get('forecastingAlgorithm')
            P = j.get('P')
            Q = j.get('Q')
            D = j.get('D')
            arima_model_type = j.get('arima_model_type')
            iterations = j.get('iterations')
            locationAddress = j.get("locationAddress")
            modelName = j.get("modelName")
            columnsNameList = j.get("columnsNameList")
            sheetId = j.get("workbookId")
            worksheetId = j.get("userWorksheetId")

            # for UI changes
            confIntPara = '0.95'

            forecastClass = \
                Forecasting.ForecastingModel(alpha=alpha, beta=beta, gamma=gamma, isTrending=isTrending,
                                             isSeasonal=isSeasonal,
                                             seasonalPeriodsManual=seasonalPeriodsManual,
                                             seasonalP=seasonalP, seasonalD=seasonalD,
                                             seasonalQ=seasonalQ, confIntPara=confIntPara,sparkSession = spark)
            response_data = \
                forecastClass.forecastingTimeSeries(data=data, count=count, len_type=len_type,
                                                    model_type=model_type, trendType=trendType,
                                                    seasonType=seasonType, forecastAlgorithm=forecastAlgorithm,
                                                    P=P, Q=Q, D=D, arima_model_type=arima_model_type,
                                                    iterations=iterations,
                                                    columnsNameList=columnsNameList,
                                                    locationAddress=locationAddress,
                                                    modelName=modelName,
                                                    sheetId=worksheetId)
        print("success")
    except Exception as e:
        print('exception = ' + str(e))
        response_data = str(json.dumps({'run_status': 'sorry! unable to process your request'})).encode('utf-8')
    status = '200 OK'

    return jsonify(success='success', message='ml_server_response', data=response_data)

"""handle all the request related to sentiment analysis"""

@app.route("/sentiment", methods=["POST", "GET"])
def sentimentAnalysis():
    # create the request data for sentiment analyis -
    # - divide the createData section in three part each for pretrained model, lexicon and ML model. -sahil
    response = Response(content_type="application/json")
    requestString = request.data.decode("utf-8")
    requestData = json.loads(requestString)
    sentimentType = requestData.get(PredictiveConstants.SENTIMENTTYPE)
    responseData = {}

    try:
        createRequestData = CreateRequestData()
        if (sentimentType.__eq__("Vivekn_Pretrained")):
            infoData = createRequestData.createPretrainedSentimentData(requestData)
            infoData.update({PredictiveConstants.SPARK: spark})
            responseData = ViveknPretrainedModel().sentimentAnalysis(infoData)
        elif (sentimentType.__eq__("Lexicon_Model")):
            infoData = createRequestData.createLexiconSentimentData(requestData)
            infoData.update({PredictiveConstants.SPARK: spark})
            responseData = LexiconSA().sentimentAnalysis(infoData)



        responseData["run_status"] = "success"
        print(responseData.get("run_status"))

    except Exception as e:
        print('exception is = ' + str(e))
        responseData = PredictiveExceptionHandling.exceptionHandling(e)

    finally:
        return jsonify(success='success', message='it was a success', data=responseData)



if (__name__ == '__main__'):
    app.run(host='0.0.0.0', port=3334, debug=False)
    spark.stop()
