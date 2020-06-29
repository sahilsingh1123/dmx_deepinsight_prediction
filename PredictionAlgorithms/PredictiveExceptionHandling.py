# this class is designed for handling exception on various errors

class PredictiveExceptionHandling():
    defaultException = ''
    exceptionMessage = ''

    # list of exceptions
    # 'Unable to infer schema for Parquet. It must be specified manually.;'
    # 'requirement failed: The input column indexed_Exhaust_Gas_Bypass_Valve_Position should have at least two distinct values.'
    @staticmethod
    def exceptionHandling(exception):
        defaultException = str(exception)
        if (defaultException.startswith("'Unable to infer schema for Parquet")):
            exceptionMessage = "Dataset not found."

        elif (defaultException.endswith("should have at least two distinct values.'")):
            defaultException = defaultException.replace("indexed_", "", 1)
            exceptionMessage = defaultException

        elif (defaultException.__contains__("requirement failed: BLAS.dot(x: Vector, y:Vector) was given Vectors with non-matching sizes:")):
            exceptionMessage = "The number of given categories is not equal to the number of categories you had selected while building a model."

        else:
            exceptionMessage = defaultException

        responseData = {'run_status': exceptionMessage}

        return responseData
