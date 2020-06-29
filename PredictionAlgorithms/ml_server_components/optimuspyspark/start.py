import collections
import copy
import dateparser
import json
import matplotlib as mpl
import pyspark.sql.functions as DataframeFunction
from OptimusSpark import ConfigureActions
from OptimusSpark import DIOptimus
from flask import Flask
from flask import Response
from flask import jsonify
from flask import request
from functools import reduce
from operator import itemgetter
from operator import or_
from optimus.ml import distancecluster as dc
from optimus.ml import keycollision as keyCol
from optimus.profiler.profiler import Profiler
from pyspark.sql.functions import *
from pyspark.sql.functions import lit
from pyspark.sql.types import TimestampType, StringType, DateType

mpl.use("TkAgg")

# https://stackoverflow.com/questions/49836676/error-after-upgrading-pip-cannot-import-name-main
#
# print(SparkSession)
# print(Optimus)

app = Flask(__name__)

def cacheTable(datasetLocation, datasetName):
    print("Caching Request for dataset {}".format(datasetName))
    tables = DIOptimus.getmanager(None).optimus.spark.catalog.listTables()
    print(tables)
    tableExists = tContains(tables, datasetName)
    if tableExists:
        df = DIOptimus.getmanager(None).optimus.spark.sql("select * from " + datasetName)
        return df
    else:
        df = DIOptimus.getmanager(None).optimus.load.parquet(datasetLocation)
        df.show()
        df.createOrReplaceTempView(datasetName)
        return df


def uncacheTable(datasetName):
    Manager = DIOptimus.getmanager(None)
    Manager.optimus.spark.catalog.uncacheTable(datasetName);
    Manager.optimus.spark.catalog.dropTempView(datasetName);


def tContains(tables, datasetName):
    for table in tables:
        if table.name.upper() == datasetName.upper():
            return True
    return False


@app.route("/configure", methods=["POST", "GET"])
def optimusHandlerConfigure():
    try:
        Manager = DIOptimus.getmanager(sparkURL = request.data.decode("utf-8"))
        return jsonify(status="success", message="configuration Succesfull")
    except Exception as e:
        print(e)
        return jsonify(status="failed", message=str(e)[:200]+"..")


@app.route("/refresh", methods=["POST", "GET"])
def refreshDataset():
    requestString = request.data.decode("utf-8")
    requestData = json.loads(requestString)
    info = requestData["info"]
    datasetName = info.get("recipeDatasetName")
    fileLocation = info.get("recipeFileLocation")
    hdfsLocation = info.get("hdfsLocation")
    steps = requestData["data"]
    finalDataset = datasetName+"_FINAL.parquet"
    try:
        for step in steps:
            optimusOperations(step, True)

        if len(steps) > 0:
            Manager = DIOptimus.getmanager(None)
            df = cacheTable(None, datasetName)
            DIOptimus.getmanager(None).optimus.spark.sql("REFRESH TABLE "+datasetName)
            df = df.drop("__ID__")
            print("----  Count {}".format(df.count()))
            df.write.parquet(hdfsLocation+finalDataset)
            uncacheTable(datasetName)


        return jsonify(status="success", message="success",location=datasetName+"_FINAL")


    except Exception as e:
        print(e)
        uncacheTable(datasetName)
        return jsonify(status="failed", message=str(e)[:200] + "..")


@app.route("/optimus", methods=["POST", "GET"])
def optimusHandler():
    try:
        return optimusOperations(None)
    except Exception as e:
        print(e)
        return jsonify(status="failed", message=str(e)[:200]+"..")


def optimusOperations(recipe, refresh=False):
    response = Response(content_type="application/json")
    requestData = None
    if recipe is not None:
        requestString = str(recipe)
        requestData = recipe
    else:
        requestString = request.data.decode("utf-8")
        requestData = json.loads(requestString)
    print("Request  data ", requestData)
    Manager = DIOptimus.getmanager(None)
    hdfsLocation = requestData.get('hdfsLocation')
    recipeInfo = requestData.get('recipeInfo')

    print('----------------  HDFS LOCATION --------------', hdfsLocation)

    if requestData.get("requestType") == "createToParquet":
        createFileUtil = ConfigureActions.CreateFileUtil(requestData)
        result = createFileUtil.convertToParquet()
        if result == "success":
            res = jsonify(status="success", message="Parquet file create successfully", data="")
        else:
            res = jsonify(status="failed", message="exception occurred while creating the parquet file", data="")
        return res

    responseData = {}
    responseData["category"] = requestData.get("category")

    datasetName = requestData.get("datasetName")
    datasetName = datasetName.upper()
    print("Dataset Name ", datasetName)
    tempName = recipeInfo.get("recipeDatasetName")
    newDatasetName = hdfsLocation + tempName + '.parquet'
    print("File Location  ++++++++++++++++  ", newDatasetName)
    actions = requestData.get("actions")
    if recipe:
        df = cacheTable(newDatasetName, tempName)
    else:
        df = Manager.optimus.load.parquet(newDatasetName)

    df = df.withColumn("__ID__", monotonically_increasing_id())
    df.show(5)
    sorting = None
    if "sorting" in requestData:
        sorting = requestData.get("sorting")
    dataframe = df
    responseDataframe = None
    filterParameters = None

    if (actions is None and (requestData.get("recipeData").get("category") == "FINISH")) and (
            len(requestData.get("recipeData").get("actions")[0].get("filterParameters")) > 0):
        dataframe = prepareFilterResponse(requestData.get("recipeData"),
                                          requestData.get("recipeData").get("actions")[0].get("filterParameters"),
                                          dataframe, responseData, df=df)
    if actions is not None:
        diGsonResponse = []
        for x in actions:
            operation_type = x.get('type')
            action = x.get('action')
            parameter = x.get('parameter')
            filterParameters = x.get('filterParameters')
            filterOnlyQuery = x.get('filterOnlyQuery')
            executedQuery = False
            columnUpdated = False
            checker = None

            if requestData.get("category") == 'STATS':
                profiler = Profiler()
                columnName = requestData["eventedColumnName"]
                if (getDataTypeOf(dataframe, columnName) == "date") | (getDataTypeOf(dataframe, columnName) == "timestamp"):
                    profiledData = profiler.columns(dataframe.withColumn(columnName, dataframe[columnName].cast("string")), columnName)
                elif dataframe.filter(isnan(columnName) | col(columnName).isNull()).count() == dataframe.count() :
                    profiledData = profiler.columns(dataframe.na.fill("Null"), columnName)
                else:
                    profiledData = profiler.columns(dataframe, columnName)
                profiledData = profiledData.get("columns").get(columnName)
                profiledData.pop("dtypes_stats", None)
                profiledData.pop("hist", None)
                newcolumnname = columnName + "_length"
                if (not ((getDataTypeOf(dataframe, columnName) == "date") | (getDataTypeOf(dataframe, columnName) == "timestamp"))) and ((dataframe.filter(isnan(columnName)) or (col(columnName).isNull()).count() == dataframe.count())):
                    dataframe = dataframe.na.fill("Null")
                    dataframe = dataframe.cols.append(newcolumnname, length(columnName))
                    stats = profiler.general_stats(dataframe, newcolumnname)
                    extrastatus = profiler.extra_numeric_stats(dataframe, newcolumnname, stats=stats, relative_error=1)
                else:
                    dataframe = dataframe.cols.append(newcolumnname, length(columnName))
                    stats = profiler.general_stats(dataframe, newcolumnname)
                    extrastatus = profiler.extra_numeric_stats(dataframe, newcolumnname, stats=stats, relative_error=1)
                finalStatus = dict()
                finalStatus["min"] = stats.get(newcolumnname)["min"]
                finalStatus["max"] = stats.get(newcolumnname)["max"]
                finalStatus["stddev"] = stats.get(newcolumnname)["stddev"]
                finalStatus["mean"] = stats.get(newcolumnname)["mean"]
                finalStatus["sum"] = stats.get(newcolumnname)["sum"]
                finalStatus["variance"] = stats.get(newcolumnname)["variance"]
                finalStatus["zeros"] = stats.get(newcolumnname)["zeros"]
                finalStatus["zeros"] = stats.get(newcolumnname)["zeros"]
                finalStatus["median"] = extrastatus["median"]
                finalStatus["range"] = extrastatus["range"]
                finalStatus['average'] = dataframe.cols._exprs([avg], newcolumnname)
                total_count = dataframe.cols._exprs([count], newcolumnname)
                finalStatus['count'] = total_count
                finalStatus['p_count'] = (int(total_count) / int(total_count)) * 100

                responseData['columnName'] = columnName
                responseData['stats'] = finalStatus
                responseData['profile'] = profiledData
                dataframe.cols.drop(newcolumnname)
                dataframe = dataframe.drop("__ID__")
                return jsonify(status="success", message="It was a success", data=responseData)
            if operation_type == "cluster":
                dfc = dataframe_1 = Manager.optimus.load.parquet(newDatasetName)
                if not parameter.get('keying'):
                    dfc = dc.levenshtein_cluster(dataframe_1, requestData["eventedColumnName"])
                else:
                    if 'ngramSize' in parameter:
                        dfc = getattr(keyCol, parameter.get('keying'))(dataframe_1, requestData["eventedColumnName"],
                                                                       int(parameter.get('ngramSize')))
                    else:
                        dfc = getattr(keyCol, parameter.get('keying'))(dataframe_1, requestData["eventedColumnName"])
                dfc.show()
                prepareClusterResponse(dfc, responseData)
                res = jsonify(status="success", message="It was a success", data=responseData)
                return res
            elif action == "min-max" or action == "min-max-time":
                facetData = {}
                dataframe = df
                print(*parameter)
                if "eventedColumnName" in requestData:
                    eventedColumnName = requestData["eventedColumnName"]
                else:
                    eventedColumnName = parameter
                min = dataframe.cols.min(eventedColumnName)
                max = dataframe.cols.max(eventedColumnName)
                responseData["data"] = {}
                min_max_data = {}
                min_max_data[eventedColumnName] = []
                if action == "min-max-time":
                    min_max_data[eventedColumnName].append({"min": str(min), "max": str(max)})
                else:
                    min_max_data[eventedColumnName].append({"min": min, "max": max})
                facetData["data"] = min_max_data
                facetData["count"] = dataframe.count()
                if requestData.get("category") is None:
                    facetData["category"] = action
                else:
                    facetData["category"] = requestData.get("category")
                diGsonResponse.append(copy.deepcopy(facetData))
                checker = True
            elif action == 'mismatch':
                print("hello mismatch")
                parameterData = ""
                if isinstance(parameter, list):
                    parameterData = parameter[0]
                else:
                    parameterData = parameter
                dataframe = dataframe.withColumn("__Dmx_Internal_MISmAtCh" +parameterData,when(col(parameterData).isNull(),"Null/Invalid")
                                                .when(regexp_extract(col(parameterData), ".?[0-9]+", 0) == "", "String")
                                                 .when(regexp_extract(col(parameterData),"^[0-9]\d*(\.\d+)?$",0) != "", "Numbers")
                                                 .otherwise("AlphaNumeric"))
                dataframe.show(10)
                facetData = {}
                # dataframe = df
                # if filterParameters is not None:
                #     dataframe = applyFilter(dataframe, requestData, filterParameters)

                data = dataframe.cols.frequency("__Dmx_Internal_MISmAtCh" +parameterData, 1000)
                data = collections.OrderedDict(sorted(data.items(), key=itemgetter(0), reverse=True))
                jsonData = {}
                jsonData[parameterData] = data.get("__Dmx_Internal_MISmAtCh" +parameterData)
                facetData["data"] = jsonData
                facetData["count"] = len(data)
                facetData["category"] = requestData.get("category")
                diGsonResponse.append(copy.deepcopy(facetData))
                checker = True
                #dat.show(20,dataframe False)
                # dataframe.drop("newColumn")
            elif action == "frequency":
                facetData = {}
                dataframeCount = df
                if filterParameters is not None:
                    dataframeCount = applyFilter(dataframe, requestData, filterParameters)

                dataframeCount = dataframeCount.cols.frequency(parameter, 1000)
                dataframeCount = collections.OrderedDict(sorted(dataframeCount.items(), key=itemgetter(0), reverse=True))
                facetData["data"] = dataframeCount
                facetData["count"] = len(dataframeCount)
                if requestData.get("category") is None:
                    facetData["category"] = action
                else:
                    facetData["category"] = requestData.get("category")
                diGsonResponse.append(copy.deepcopy(facetData))
                checker = True
            elif action == "count-Invalid":
                facetData = {}
                dataframe = df
                if "eventedColumnName" in requestData:
                    eventedColumnName = requestData["eventedColumnName"]
                else:
                    eventedColumnName = parameter
                if filterParameters is not None:
                    dataframe = applyFilter(dataframe, requestData, filterParameters)
                facetData["data"] = {}
                total_count = dataframe.count()
                facetData["count"] = total_count
                if (getDataTypeOf(dataframe, eventedColumnName) != "date" and getDataTypeOf(dataframe, eventedColumnName) != "timestamp"):
                    dataframe = dataframe.filter(
                        (((dataframe[eventedColumnName] == "") | (dataframe[eventedColumnName].isNull())) | (
                            isnan(dataframe[eventedColumnName]))))
                else:
                    dataframe = dataframe.filter(((dataframe[eventedColumnName] == "") | (dataframe[eventedColumnName].isNull())))
                invalid_count = dataframe.count()
                invalid_count_data = {eventedColumnName: []}
                invalid_count_data[eventedColumnName].append({"totalCount": total_count, "invalidCount": invalid_count})
                facetData["data"] = invalid_count_data
                if requestData.get("category") is None:
                    facetData["category"] = action
                else:
                    facetData["category"] = requestData.get("category")
                diGsonResponse.append(copy.deepcopy(facetData))
                checker = True
            if requestData.get("category") == 'QUERY':
                try:
                    tempName = str(datasetName)[str(datasetName).index('_') + 1:]
                    sqldataframe = dataframe.limit(10)
                    sqldataframe.createOrReplaceTempView(tempName)
                    sqldataframe.sql_ctx.sql(requestData['query'])
                except Exception as e:
                    print(str(e))
                    return jsonify(status="failed", message=str(e.desc), data=responseData)
            if action == 'CHANGE_DTYPE':
                try:
                    sqldataframe = dataframe.limit(10)
                    if parameter[0] == 'date':
                        date_udf = createToDateUDF("DMY", DateType())  # udf registration
                        dataframe = dataframe.withColumn(requestData.get("eventedColumnName"), date_udf(requestData.get("eventedColumnName")))
                    elif parameter[0] == 'timestamp':
                        date_udf = createToDateUDF("DMY", TimestampType())  # udf registration
                        dataframe = dataframe.withColumn(requestData.get("eventedColumnName"), date_udf(requestData.get("eventedColumnName")))
                    else:
                        changedTypedf = sqldataframe.withColumn(requestData.get("eventedColumnName"), sqldataframe[requestData.get("eventedColumnName")].cast(*parameter))
                        del changedTypedf
                except Exception as e:
                    print(str(e))
                    return jsonify(status="failed", message=str(e.desc), data=responseData)

            if (checker == True) & (len(actions) == len(diGsonResponse)):
                if len(diGsonResponse) == 1:
                    res = jsonify(status="success", message="It was a success", data=diGsonResponse[0])
                else:
                    res = jsonify(status="success", message="It was a success", data=diGsonResponse)
                return res
            if checker == True:
                continue

            if not refresh:
                tempName = recipeInfo.get("recipeDatasetName")
                newDatasetdf = Manager.optimus.load.parquet(hdfsLocation + tempName + '_preview.parquet')
                newDatasetName2 = hdfsLocation + tempName + '_undo.parquet'
                newDatasetdf = newDatasetdf.drop("__ID__")
                newDatasetdf.write.parquet(newDatasetName2, mode="overwrite")

            if action == "drop" and operation_type == "rows":
                print(*parameter)
                uParameter = []
                for action in parameter:
                    uParameter.append(eval(action))
                parameter = uParameter
            elif action == "sort":
                f = getattr(dataframe, operation_type)
                dataframe = getattr(f, action)(*parameter)
                prepareResponse(dataframe, responseData)
            if operation_type == "cols" and action == "null":
                colname = parameter[0]
                dataframe = dataframe.withColumn(colname, lit(""))
                dataframe = getattr(dataframe, action)(*parameter)
            if (operation_type == "None") & (action != 'None'):
                if action == 'limit':
                    if parameter[1] == 'top':
                        parameter = int(parameter[0])
                        dataframe = getattr(dataframe, action)(parameter)
                    elif parameter[2] == 'range':
                        parameter1 = int(parameter[0])
                        data_from = dataframe.limit(parameter1)
                        parameter2 = int(parameter[1])
                        data_to = dataframe.limit(parameter2)
                        data_range = data_to.exceptAll(data_from)
                        prepareResponse(data_range, responseData)
                elif action == 'null':
                    colname = parameter[0]
                    dataframe = dataframe.withColumn(colname, regexp_replace(col(colname), '.+', None))
                elif action == 'null2':
                    colname = parameter[0]
                    dataframe = dataframe.withColumn(colname, regexp_replace(col(colname), '.+', ""))
                elif action == 'trim_length':
                    dataframe = dataframe.withColumn(parameter[0], col(parameter[0]).substr(1, parameter[1]))
                elif action == 'reorder':
                    dataframe = dataframe.select(parameter)
                elif action == 'to_date':
                    eventedColumn = requestData["eventedColumnName"]
                    if filterOnlyQuery and recipeInfo.get("isInitialized") and filterParameters is not None:
                        dataframe = prepareFilterResponse(requestData, filterParameters, dataframe, responseData, df=df)
                        dataframe = dataframe.limit(1000)
                        executedQuery = True
                    if parameter[1] == None:
                        local = "DMY"
                    else:
                        local = parameter[1]
                    if parameter[2] == "TimeStamp":
                        date_udf = createToDateUDF(local, TimestampType())  # udf registration
                    else:
                        date_udf = createToDateUDF(local, DateType())  # udf registration

                    dataframe = dataframe.withColumn("New_" + eventedColumn, date_udf(eventedColumn))
                    dataframe = dataframe.cols.move("New_" + eventedColumn,"after",parameter[0])
                else:
                    dataframe = getattr(dataframe, action)(*parameter)
            elif requestData.get("category") == 'QUERY':
                tempName = str(datasetName)[str(datasetName).index('_') + 1:]
                dataframe.createOrReplaceTempView(tempName)
                if filterOnlyQuery and recipeInfo.get("isInitialized") and filterParameters is not None:
                    dataframe = prepareFilterResponse(requestData, filterParameters, dataframe, responseData, df=df)
                    executedQuery = True
                dataframe = dataframe.sql_ctx.sql(requestData['query'])
                prepareResponse(dataframe, responseData)
                dataframe = dataframe.drop("__ID__")
                responseData["count"] = dataframe.count()
                responseData['status'] = "success"
                responseData['message'] = "It was a success"
            elif action == 'CHANGE_DTYPE':
                try:
                    dataframe = dataframe.withColumn(requestData.get("eventedColumnName"), dataframe[requestData.get("eventedColumnName")].cast(*parameter))
                except Exception as e:
                    print(str(e))
                    return jsonify(status="failed", message=str(e.desc), data=responseData)
            elif operation_type == "cols" and action == 'nest':
                dataframe = dataframe.cols.nest(parameter[0], parameter[1], shape="string", separator=parameter[2])
                if parameter[3] != "" or parameter[4] != "":
                    dataframe = dataframe.withColumn(parameter[1],
                                                     concat(lit(parameter[3]), (col(parameter[1])), lit(parameter[4])))
            elif operation_type == "cols" and action == 'unnest':
                # datatypes = dataframe.dtypes()
                retainColumn = parameter[3]
                parameter.pop(3)
                eventedColumnName = requestData["eventedColumnName"]
                old_columns = dataframe.columns
                f = getattr(dataframe, operation_type)
                print(f)
                dataframe = getattr(f, action)(*parameter)
                new_columns = dataframe.columns
                diff_cols = list([item for item in new_columns if
                                  item not in old_columns])  # list(set(set(new_columns) - set(old_columns)))

                for i in reversed(diff_cols):
                    print(i)
                    f = getattr(dataframe, operation_type)
                    params = [i, "after", eventedColumnName]
                    dataframe = getattr(f, "move")(*params)
                if retainColumn == False:
                    dataframe = dataframe.cols.drop(eventedColumnName)
                    columnUpdated = True
            elif operation_type == 'cols' and action == 'trim':
                selectedColumn = parameter[0]
                dataframe = dataframe.withColumn(selectedColumn, (trim(col(selectedColumn))))
            elif operation_type == 'cols' and action == 'fill_na':
                eventedColumnName = parameter[0]
                if (dataframe.schema[eventedColumnName].dataType) != DateType:
                    f = getattr(dataframe, operation_type)
                    print(*parameter)
                    dataframe = getattr(f, action)(*parameter)
                else:
                    dataframe = withColumn(eventedColumnName, col(eventedColumnName).cast(StringType))
                    f = getattr(dataframe, operation_type)
                    dataframe = getattr(f, action)(*parameter)
                    dataframe = dataframe.withColumn(eventedColumnName, col(eventedColumnName).cast(TimestampType))

            elif action == 'SPLIT_DATE_TIME':
                eventedColumnName = requestData.get("eventedColumnName")
                old_columns = dataframe.columns
                if parameter[0] == 'DateTime':
                    dataframe = dataframe.withColumn("date_" + eventedColumnName, to_date(col(eventedColumnName)))
                    dataframe = dataframe.withColumn("time_" + eventedColumnName,
                                                     date_format(col(eventedColumnName), "h:m:s a"))

                elif parameter[0] == 'Individual':
                    for subfunction in parameter[1]:
                        if(subfunction == 'year'):
                            dataframe = dataframe.withColumn("Year_"+eventedColumnName,year(eventedColumnName))
                        if(subfunction == 'month'):
                            dataframe = dataframe.withColumn("Month_"+eventedColumnName,month(eventedColumnName))
                        if(subfunction == 'day'):
                            dataframe = dataframe.withColumn("Day_"+eventedColumnName,dayofmonth(eventedColumnName))
                        if(subfunction == 'hour'):
                            dataframe = dataframe.withColumn("Hours_" + eventedColumnName,
                                                             date_format(col(eventedColumnName), "h"))
                        if(subfunction == 'minutes'):
                            dataframe = dataframe.withColumn("Minutes_" + eventedColumnName,
                                                             date_format(col(eventedColumnName), "m"))
                        if(subfunction == 'seconds'):
                            dataframe = dataframe.withColumn("Seconds_" + eventedColumnName,
                                                             date_format(col(eventedColumnName), "s"))
                    new_columns = dataframe.columns
                    diff_cols = list([item for item in new_columns if
                                      item not in old_columns])  # list(set(set(new_columns) - set(old_columns)))
                    for i in reversed(diff_cols):
                        print(i)
                        f = getattr(dataframe, operation_type)
                        params = [i, "after", eventedColumnName]
                        dataframe = getattr(f, "move")(*params)

            elif action == 'SPLIT_BY_LENGTH':
                eventedColumnName = requestData.get("eventedColumnName")
                old_columns = dataframe.columns
                dataframe = dataframe.withColumn("Col1_" + eventedColumnName,
                                                     substring(eventedColumnName, 1, int(parameter[0])))
                udf1 = udf(lambda x: x[int(parameter[0]):], StringType())
                dataframe = dataframe.withColumn("Col2_" + eventedColumnName, udf1(eventedColumnName))
                new_columns = dataframe.columns
                diff_cols = list([item for item in new_columns if
                                  item not in old_columns])  # list(set(set(new_columns) - set(old_columns)))
                for i in reversed(diff_cols):
                    print(i)
                    f = getattr(dataframe, operation_type)
                    params = [i, "after", eventedColumnName]
                    dataframe = getattr(f, "move")(*params)

            elif operation_type == 'cols' and action == 'collapse_spaces':
                selectedColumn = parameter[0]
                dataframe = df = dataframe.withColumn(selectedColumn,
                                                      regexp_replace(col(selectedColumn), '[\\s]+', " "))
            elif operation_type == 'cols' and action == 'remove_quotes':
                selectedColumn = parameter[0]
                string = "[\"]+"
                dataframe = dataframe.withColumn(selectedColumn, regexp_replace(col(selectedColumn), string, ""))
            elif operation_type == 'cols' and action == 'remove_symbols':
                selectedColumn = parameter[0]
                string = "[^0-9a-zA-Z" + parameter[1] + "]+"
                dataframe = dataframe.withColumn(selectedColumn, regexp_replace(col(selectedColumn), string, ' '))
            elif operation_type == 'cols' and action == 'replace':
                selectedColumn = parameter[0]
                string = "" + parameter[1] + ""
                replace_string = parameter[2]
                # string = "[^0-9a-zA-Z" + parameter[1] + "]+"
                dataframe = dataframe.withColumn(selectedColumn,
                                                 regexp_replace(col(selectedColumn), string, replace_string))
            elif operation_type == 'cols' and action == 'duplicate_col':
                eventedColumnName = requestData["eventedColumnName"]
                dataframe = dataframe.withColumn(eventedColumnName + "_copy", col(eventedColumnName))
                dataframe.show(2)
                dataframe = dataframe.cols.move(eventedColumnName + "_copy", "after", eventedColumnName)
            elif operation_type == 'rows' and action == 'drop_rows':
                # eventedColumnName = parameter[0]             //Previous logic
                # dataframe = dataframe.rows.drop((col(eventedColumnName) == parameter[1]))
                try:
                    if len(parameter) == 2 and parameter[1] == True:
                        eventedColumnName = parameter[0]
                        dataframe = dataframe.where(col(eventedColumnName).isNotNull())                    # '!(currency == \'Dollars\') OR currency is null'
                    else:
                        if parameter[2] == False:
                            querystring = '!(' + parameter[1] + ') OR ' + parameter[0] + ' is null'
                        elif parameter[2] == True:
                            querystring = '!(' + parameter[1] + ')'

                        # somehow nulls get filtered out regardless of th query type. We need to retain them ...
                        # if (('null' not in parameter[1]) or ('NULL' not in parameter[1])):
                        # querystring = '(' + eventedColumnName + ' is null) OR ' + querystring

                        # 'currency!=\'Shillings\' OR currency is null'
                        # 'dish_count > 100 OR dish_count is null'

                        dataframe = dataframe.filter(querystring)
                    # dataframe = dataframe.rows.drop(querystring)
                except ValueError:
                       res = jsonify(status="Failed", message=ValueError[30:])

            elif operation_type == "cols" and action == 'rename':
                dataframe = dataframe.cols.rename(*parameter)
                columnUpdated = True
            elif operation_type == "cols" and action == 'drop':
                dataframe = dataframe.cols.drop(*parameter)
                columnUpdated = True;
            elif operation_type == "FILTER":
                dataframe = prepareFilterResponse(requestData, filterParameters, dataframe, responseData, df=df)
            elif action == "EDIT_CELL":
                row = parameter["row"]
                print(row)
                eventedColumnName = parameter["eventedColumnName"]
                oldValue = parameter["oldValue"]
                newValue = parameter["value"]
                print(df["__ID__"] == row)

                print("//////////////////////////////////////////////////////")
                print('Filter Parameter  ', filterParameters)
                print("//////////////////////////////////////////////////////")

                whenClause = getWhenCaluse(dataframe, filterParameters)

                if whenClause is not None:
                    sparkFunction = x["sparkFunction"]
                    responseDataframe = dataframe
                    responseDataframe = prepareFilterResponse(requestData, filterParameters, responseDataframe,responseData, df=df)
                    responseDataframe = responseDataframe.withColumn(eventedColumnName,when(((whenClause) & ((df["__ID__"] == row) & (((df[eventedColumnName] == oldValue) | (df[eventedColumnName].isNull())) | (length(df[eventedColumnName]) == 0)))),newValue).otherwise(df[eventedColumnName]))
                    prepareResponse(responseDataframe, responseData)
                    responseData["count"] = responseDataframe.count()
                    dataframe = dataframe.withColumn(eventedColumnName, when(((whenClause) & ((df["__ID__"] == row) & (
                            ((df[eventedColumnName] == oldValue) | (df[eventedColumnName].isNull())) | (
                            length(df[eventedColumnName]) == 0)))), newValue).otherwise(df[eventedColumnName]))
                else:
                    dataframe = dataframe.withColumn(eventedColumnName, when(((df["__ID__"] == row) & (
                            ((df[eventedColumnName] == oldValue) | (df[eventedColumnName].isNull())) | (
                            length(df[eventedColumnName]) == 0))), newValue).otherwise(df[eventedColumnName]))
            elif action == "EDIT_ALL_OCCURRENCES":
                eventedColumnName = parameter["eventedColumnName"]
                oldValue = parameter["oldValue"]
                newValue = parameter["value"]
                print(eventedColumnName)
                whenClause = getWhenCaluse(dataframe, filterParameters)
                if whenClause is not None:
                    sparkFunction = x["sparkFunction"]
                    responseDataframe = dataframe
                    responseDataframe = prepareFilterResponse(requestData, filterParameters, responseDataframe,
                                                              responseData, df=df)
                    if oldValue == "(NULL)":
                        responseDataframe = responseDataframe.withColumn(eventedColumnName, when(((whenClause) & ((df[eventedColumnName].isNull()) | (length(df[eventedColumnName]) == 0))), newValue).otherwise(
                            df[eventedColumnName]))
                    else:
                        responseDataframe = responseDataframe.withColumn(eventedColumnName, when(((whenClause) & (df[eventedColumnName] == oldValue)),newValue).otherwise(df[eventedColumnName]))

                    prepareResponse(responseDataframe, responseData)
                    responseData["count"] = responseDataframe.count()

                    if oldValue == "(NULL)":
                        dataframe = dataframe.withColumn(eventedColumnName, when(((whenClause) & ((df[eventedColumnName].isNull()) | (length(df[eventedColumnName]) == 0))), newValue).otherwise(
                            df[eventedColumnName]))
                    else:
                        dataframe = dataframe.withColumn(eventedColumnName, when(((whenClause) & (df[eventedColumnName] == oldValue)),newValue).otherwise(df[eventedColumnName]))
                else:
                    if oldValue == "(NULL)":
                        dataframe = dataframe.withColumn(eventedColumnName, when(((df[eventedColumnName].isNull()) | (length(df[eventedColumnName]) == 0)),newValue).otherwise(df[eventedColumnName]))
                    else:
                        dataframe = dataframe.withColumn(eventedColumnName, when((df[eventedColumnName] == oldValue),newValue).otherwise(df[eventedColumnName]))
            elif action == "EDIT_ALL_OCCURRENCES_CLUSTER":
                whenClause = getWhenCaluse(dataframe, filterParameters)
                if whenClause is not None:
                    response_dataframe = dataframe
                    response_dataframe = prepareFilterResponse(requestData, filterParameters, response_dataframe,
                                                               responseData, df=df)
                    response_dataframe = replaceClusterData(response_dataframe, parameter)
                    prepareResponse(response_dataframe, responseData)
                    responseData["count"] = response_dataframe.count()
                    dataframe = replaceClusterData(dataframe, parameter)
                else:
                    dataframe = replaceClusterData(dataframe, parameter)
            else:
                eventedColumnName = requestData["eventedColumnName"]
                whenClause = getWhenCaluse(dataframe, filterParameters)
                if (whenClause is not None) and ("sparkFunction" in x):
                    sparkFunction = x["sparkFunction"]
                    responseDataframe = dataframe
                    responseDataframe = prepareFilterResponse(requestData, filterParameters, responseDataframe,
                                                              responseData, df=df)
                    if 'move' != sparkFunction:
                        responseDataframe = responseDataframe.withColumn(eventedColumnName,
                                                                         when(whenClause,
                                                                              getattr(DataframeFunction, sparkFunction)(
                                                                                  dataframe[eventedColumnName])).otherwise(
                                                                             dataframe[eventedColumnName]))
                    prepareResponse(responseDataframe, responseData)
                    responseData["count"] = responseDataframe.count()
                    if 'move' != sparkFunction:
                        dataframe = dataframe.withColumn(eventedColumnName,
                                                         when(whenClause, getattr(DataframeFunction, sparkFunction)(
                                                             dataframe[eventedColumnName])).otherwise(
                                                             dataframe[eventedColumnName]))

                    if 'move' == sparkFunction:
                        f = getattr(dataframe, operation_type)
                        dataframe = getattr(f, action)(*parameter)
                        prepareResponse(dataframe, responseData)
                elif action == 'initcap':
                    dataframe = dataframe.withColumn(eventedColumnName, initcap(col(eventedColumnName)))
                else:
                    if (operation_type != 'NONE') & (action != 'NONE'):
                        f = getattr(dataframe, operation_type)
                        print(*parameter)
                        dataframe = getattr(f, action)(*parameter)
                    else:
                        dataframe = dataframe.count()

    print(type(dataframe))

    if (isinstance(dataframe, int)):
        print("Count  ", dataframe)
    elif isinstance(dataframe, dict):
        dataframe = collections.OrderedDict(sorted(dataframe.items(), key=itemgetter(0), reverse=True))
        responseData["data"] = dataframe
        responseData["count"] = len(dataframe)
        print(dataframe)
    else:
        limit = None
        tempName = recipeInfo.get("recipeDatasetName")
        newDatasetNameTemp = hdfsLocation + tempName + '_temp.parquet'
        if refresh:
            dataframe.createOrReplaceTempView(tempName)
            dataframe.show()
            return

        if (recipeInfo.get("isInitialized")):
            limit = 1000
            if filterParameters is not None and not executedQuery and not columnUpdated:
                dataframe = prepareFilterResponse(requestData, filterParameters, dataframe, responseData, df=df)
            newDatasetName = hdfsLocation + tempName + '_preview.parquet'
        else:
            newDatasetName = hdfsLocation + tempName + '.parquet'
        dataframe = dataframe.drop("__ID__")
        dataframe.write.mode("overwrite").parquet(newDatasetNameTemp)
        dataframe.unpersist()
        newDF = Manager.optimus.load.parquet(newDatasetNameTemp)
        if sorting is not None:
            newDF = newDF.rows.sort(*sorting)
        if limit is not None:
            newDF = newDF.limit(1000)
        if not refresh:
            newDF.write.mode("overwrite").parquet(newDatasetName)
            newDF.show(5);
        print("New Dataset Name  ", newDatasetName)
        responseData["newDatasetName"] = newDatasetName
        responseData["count"] = newDF.count()
        if refresh:
            newDF.createOrReplaceTempView(tempName)

        print("Total Recored Count  " + str(responseData["count"]))
    df.unpersist()
    res = jsonify(status="success", message="It was a success", data=responseData)
    return res

def iterateOverDataFrame(dataframe):
    i = 0
    for x in dataframe.rdd.collect():
        print(x)
        i += 1
        if i == 5:
            break


def filterQuery(filters, eventedColumnName, dataframe, forFrequency=False):
    queries = []
    print(filters)
    for x in filters.keys():
        if forFrequency and eventedColumnName != None and x == eventedColumnName:
            continue
        column_name = filters[x].get('columnName')
        selected_value = filters[x]
        if not selected_value:
            continue

        facetType = selected_value.get('type')
        data = selected_value.get('value')

        if 'TEXT_FILTER' in facetType:
            data = dict(data)
            query_string = column_name
            if data.get("caseSensitive"):
                query_string += ""

            if data.get("regex"):
                query_string += " rlike "
            elif data.get("isNot"):
                if data.get("characterSet") == "IsExactly":
                    query_string += " <> "
                elif data.get("characterSet") == "Contains":
                    query_string += " not like "
            elif not data.get("isNot"):
                if data.get("characterSet") == "IsExactly":
                    query_string += " = "
                elif data.get("characterSet") == "Contains":
                    query_string += " like "



            if "characterSet" in data:
                if data.get("characterSet") == "IsExactly":
                    if 'valuea' in data:
                        string1 = ""
                        if data.get("regex"):
                            string1 = data.get("valuea")
                            string1 = string1.replace("\\","\\\\\\")
                            query_string += "\"" + string1 + "\" "
                        else:
                            query_string += "\"" + data.get("valuea") + "\" "
                elif data.get("characterSet") == "Contains":
                    if ("isContains" in data) & (data.get('isContains') == "any"):
                        query_string += "\"%" + data.get("valuea") + "%\" "
                    elif ("isContains" in data) & (data.get('isContains') == "startsWith"):
                        query_string += "\"" + data.get("valuea") + "%\" "
                    elif ("isContains" in data) & (data.get('isContains') == "endsWith"):
                        query_string += "\"%" + data.get("valuea") + "\" "
            if query_string != column_name:
                queries.append(query_string)
        elif 'TEXT' in facetType:
            query_string = "" + column_name + " in ("
            for y in data:
                if (y == 'null'):
                    query_string_temp = column_name + " is null "
                    if (len(data) > 1):
                        query_string_temp += " OR " + query_string
                    query_string = query_string_temp
                else:
                    if str(dataframe.schema[column_name].dataType) != "BooleanType":
                        query_string += "'" + y + "',"
                    else:
                        query_string += "" + y + ","
            position = query_string.rfind(',')
            if (position != -1):
                query_string = query_string[:position]
                query_string += ")"
            queries.append('(' + query_string + ')')
        elif 'NUMERIC' in facetType:
            print(dict(data).get("max"))
            if (dict(data).get("max") is not None) & (dict(data).get("min") is not None):
                query_string = "" + column_name + " < " + str(
                    dict(data).get("max")) + " and " + column_name + " > " + str(
                    dict(data).get('min'))
                queries.append(query_string)
        elif 'TIMELINE' in facetType:
            if 'max' in dict(data) and 'min' in dict(data):
                query_string = "DATE(" + column_name + ") < DATE(" + "\"{}\"".format(str(
                    dict(data).get("max"))) + ") and DATE(" + column_name + ") > DATE(" + "\"{}\"".format(str(
                    dict(data).get('min'))) + ")"
                queries.append(query_string)
                print(query_string)
        elif 'INVALID' in facetType:
            if 'show' in dict(data):
                getDataOf = dict(data).get('show')
                if getDataOf == 'Valid':
                    query_string = "" + column_name + " is not NULL or " + column_name + " != \"\""
                elif getDataOf == 'InValid':
                    query_string = "(" + column_name + " is NULL or " + column_name + " = \"\")"
                else:
                    query_string = ""

                if query_string != "":
                    queries.append(query_string)
                    print(query_string)
        elif 'MISMATCH' in facetType:
            print("hello mismatch")
            dataframe.show(5)
            query_string = "" + "Types_in_" +column_name + " in ("
            for y in data:
                if (y == 'null'):
                    query_string_temp =  "" + "__Dmx_Internal_MISmAtCh" + column_name + " is null"
                    if (len(data) > 1):
                        query_string_temp += " OR " + query_string
                    query_string = query_string_temp
                else:
                    if str(dataframe.schema["__Dmx_Internal_MISmAtCh" + column_name].dataType) != "BooleanType":
                        query_string += "'" + y + "',"
                    else:
                        query_string += "" + y + ","
            position = query_string.rfind(',')
            if (position != -1):
                query_string = query_string[:position]
                query_string += ")"
            queries.append('(' + query_string + ')')

    final_query = ""
    print(queries)
    for x in queries:
        final_query += x + " and "
    if final_query:
        position = final_query.rindex("and")
        final_query = final_query[:position].strip()
    print('final_query  ', final_query)
    return final_query


def prepareResponse(dataframe, responseData):
    column_names = []
    rows = []
    schema = dataframe.columns
    i = 0
    for column in schema:
        column_name = {}
        column_name["id"] = i
        column_name["name"] = column
        column_name["field"] = i
        column_name["dataType"] = getDataTypeOf(dataframe, column)
        column_names.append(column_name)
        i += 1

    responseData["head"] = column_names

    try:
        for x in dataframe.rdd.toLocalIterator():
            i = 0
            row = {}
            for y in x:
                row[i] = y if ((y is not 'null') & (y is not None)) else '(NULL)'
                i += 1
            rows.append(row)
    except ValueError:
        print(y, ValueError)
    responseData["data"] = rows

def getDataTypeOf(dataframe, columnName):
    ddTypes = dataframe.dtypes
    for a, b in ddTypes:
        if (a == columnName):
            return b


def updateFrequency(df, dataset, selectedColumns, selectedKey,  facetColumns, response={}, filters={}):
    print("selected Columns  ", selectedColumns)
    print("columns in facet  ", facetColumns)
    print("Filter dataset  count  ", dataset.count())
    frequncies = []
    for columnNames in facetColumns:
        columnKeys = columnNames.split("@@$@@")
        column = columnKeys[0]
        facetKey = columnKeys[1]
        if (column != selectedColumns) or (facetKey != selectedKey):
            if facetKey == 'TIMELINE_FACET' or facetKey == 'NUMERIC_FACET':
                min = df.cols.min(column)
                max = df.cols.max(column)
                min_max_data = {}
                min_max_data['facetType'] = facetKey
                min_max_data[column] = []
                min_max_data['columnName'] = column
                min_max_data[column].append({"min": str(min), "max": str(max)})
                print(min_max_data)
                frequncies.append(min_max_data)
            elif facetKey == 'INVALID_FACET':
                total_count = dataset.count()
                if (getDataTypeOf(dataset, column) != "date"):
                    dataset = dataset.filter((((dataset[column] == "") | (dataset[column].isNull())) | (isnan(dataset[column]))))
                else:
                    dataset = dataset.filter(((dataset[column] == "") | (dataset[column].isNull())))
                invalid_count = dataset.count()
                invalid_count_data = {column: []}
                invalid_count_data[column].append({"totalCount": total_count, "invalidCount": invalid_count})
                invalid_count_data['facetType'] = facetKey
                invalid_count_data['columnName'] = column
                print(invalid_count_data)
                frequncies.append(invalid_count_data)
            else:
                distinctValuesDF = df.select(column).distinct()
                jsonData = []
                jsonValues = []
                for Row in distinctValuesDF.collect():
                    jsonData.append({"value": Row[0], "count": 0})
                json = dataset.cols.frequency(column)
                for data in jsonData:
                    for x in json.get(column):
                        if x.get("value") == data.get("value"):
                            data["count"] = x.get("count")
                json[column] = jsonData
               # print("Json Object ", json)
                frequncies.append({'frequency': json, 'facetType': facetKey, 'columnName':column})

    response["frequency"] = frequncies


def applyFilter(dataframe, requestData, filterParameters):
    if "eventedColumnName" not in requestData:
        eventedColumnName = None
    else:
        eventedColumnName = requestData["eventedColumnName"]

    print("~~~~~~ Filter -----   ", type(filterParameters))
    filters = filterParameters
    print("~~~~~~ Filter -----   ", filters)
    print("length of dictonary ", len(filters))

    if len(filters) != 0:
        finalQuery = filterQuery(filters, eventedColumnName, dataframe)
        print("Final Query  ", finalQuery)
        # filterqueryforfrequency = filterQuery(filters, eventedColumnName, forFrequency=True)
        # print("filter query for frequency ", filterqueryforfrequency)
        if finalQuery:
            dataframe = dataframe.filter(finalQuery)
        if "rlike" in finalQuery:
            dataframe.dataframe.filter(col(eventedColumnName).rlike())
    return dataframe


def getWhenCaluse(dataframe, filterParameters):
    con = list()
    for item in dict(filterParameters):
        print('================   ', item)
        c = None
        element = filterParameters.get(item)
        print("================   ", element)
        filterType = element.get('type')
        data = element.get('value')
        columnName = element.get("columnName")
        print(filterType)
        print(data)
        if 'TEXT' in filterType:
            for x in data:
                if c is None:
                    c = (dataframe[columnName] == str(x))
                else:
                    c = c | (dataframe[columnName] == str(x))
        elif 'NUMERIC' in filterType:
            if c is None:
                c = (dataframe[columnName] < data.get('max'))
                c = c | (dataframe[columnName] > data.get('min'))
        elif 'Invalid' in filterType:
            if c is None:
                getDataOf = data.get('show')
                if getDataOf == 'Valid':
                    c = (dataframe[columnName].isNotNull())
                    c = c | (dataframe[columnName] != "")
                elif getDataOf == 'InValid':
                    c = (dataframe[columnName].isNull())
                    c = c | (dataframe[columnName] == "")
                else:
                    c = dataframe
        con.append(c)
    finalCon = None
    for cd in con:
        if finalCon is None:
            finalCon = (cd)
        else:
            finalCon = finalCon & (cd)
    print(finalCon)

    # dataframe.withColumn(dataframe['Territory'],when((dataframe['Sales_Reason'] == "Manufacturer"),lower(dataframe['Territory'])).otherwise(dataframe['Territory'])).show()
    if finalCon is None:
        return None

    return finalCon


def prepareFilterResponse(requestData, filterParameters, dataframe, responseData, df = None):
    if "eventedColumnName" not in requestData:
        eventedColumnName = None
    else:
        eventedColumnName = requestData["eventedColumnName"]
    if "eventedKey" not in requestData:
        eventedKey = None
    else:
        eventedKey = requestData.get("eventedKey")

    facetColumns = requestData["facetColumnsforHistory"]
    print("~~~~~~ Filter -----   ", type(filterParameters))
    filters = filterParameters
    print("~~~~~~ Filter -----   ", filters)
    print("length of dictonary ", len(filters))
    # dataframe2 = dataframe
    if len(filters) != 0:
        finalQuery = filterQuery(filters, eventedColumnName, dataframe)
        print("Final Query  ", finalQuery)
        # filterqueryforfrequency = filterQuery(filters, eventedColumnName, forFrequency=True)
        # print("filter query for frequency ", filterqueryforfrequency)
        # if filterqueryforfrequency:
        #     dataframe2 = dataframe.filter(filterqueryforfrequency)
        # Treat null specially , since this is NOT a value but ABSENCE of value
        # '(event is null OR event in (\'BREAKFAST\',\'LUNCHEON\',\'DAILY MENU\',\'TIFFIN\',\'dinner\',\'MENU\',\'Breakfast\')) AND currency is null'
        if finalQuery:
            dataframe = dataframe.filter(finalQuery)

    if len(facetColumns) > 0:
        updateFrequency(df, dataframe, eventedColumnName, eventedKey, facetColumns, response=responseData, filters=filters)

    prepareResponse(dataframe, responseData)
    responseData["count"] = dataframe.count()
    print('Total Records after filter   ', responseData["count"])
    return dataframe


def prepareClusterResponse(dataframe, responseData):
    rows = []
    schema = dataframe.columns
    i = 0
    for x in dataframe.rdd.toLocalIterator():
        i = 0
        row = {}
        for y in x:
            row[schema[i]] = y
            i += 1
        rows.append(row)

    responseData["data"] = rows


def replaceClusterData(adf, parameter):
    for cluster in parameter["value"]:
        oldValue = cluster["cluster"]
        newValue = cluster["recomended"]
        eventedColumnName = parameter["eventedColumnName"]
        adf = adf.withColumn(eventedColumnName,
                             when(reduce(or_, (adf[eventedColumnName] == x for x in oldValue)), newValue).otherwise(
                                 adf[eventedColumnName]))
    return adf


def Date_formatter(date_col, Local):
    try:
        if date_col == None or date_col == "":
            return None
        # str1 = dateparser.parse(str(date_col), date_formats=['%d-%m-%Y'])
        str1 = dateparser.parse(str(date_col), settings={'DATE_ORDER': Local})
        if str1 == None:
            str1 = dateparser.parse(str(date_col))
        return str1
    except ValueError:
        print(ValueError)

def createToDateUDF(Local, DataType):
    return udf(lambda l: Date_formatter(l, Local), DataType)


if (__name__ == "__main__"):
    app.run(host='0.0.0.0', debug=False)
