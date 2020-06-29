from abc import ABC, abstractmethod

from PredictionAlgorithms.PredictiveUtilities import PredictiveUtilities


# this class will be extended with the predictive analysis class when more predictive algo will come
class PredictiveRegression(ABC):
    def __init__(self):
        pass

    def etlOperation(self, etlInfo):
        etlStats = PredictiveUtilities.performETL(etlInfo)
        return etlStats

    @abstractmethod
    def regressionEvaluation(self, regressor, regressionInfo, etlStats):
        raise NotImplementedError("subClass must implement abstract method")

    @abstractmethod
    def createGraphData(self, regressor, regressionInfo, etlStats):
        raise NotImplementedError("subClass must implement abstract method")
