from PredictionAlgorithms.PredictiveConstants import PredictiveConstants as pc


class CreateRequestData():
    def __init__(self):
        self.predictiveData = {}  # fix this thing

    def createLexiconSentimentData(self, requestData):
        self.commonForSentiment(requestData)

        positiveDatasetPath = requestData.get(pc.POSITIVEDATASETPATH)
        negativeDatasetPath = requestData.get(pc.NEGATIVEDATASETPATH)
        lemmatizedModelPath = requestData.get(pc.LEMMATIZEDPRETRAINEDMODEL)

        self.predictiveData.update({
            pc.POSITIVEDATASETPATH: positiveDatasetPath,
            pc.NEGATIVEDATASETPATH: negativeDatasetPath,
            pc.LEMMATIZEDPRETRAINEDMODEL: lemmatizedModelPath
        })

        return self.predictiveData

    def createPretrainedSentimentData(self, requestData):
        self.commonForSentiment(requestData)

        documentPretrainedPipeline = requestData.get(pc.EXPLAINDOCUMENTDL)
        viveknPretrainedModel = requestData.get(pc.VIVEKNPRETRAINEDMODEL)
        sparknlpPathMapping = {}

        self.predictiveData.update({
            pc.DOCUMENTPRETRAINEDPIPELINE: documentPretrainedPipeline,
            pc.SPARKNLPPATHMAPPING: sparknlpPathMapping,
            pc.VIVEKNPRETRAINEDMODEL: viveknPretrainedModel,
        })

        return self.predictiveData

    def commonForSentiment(self, requestData):

        fileLocation = requestData.get(pc.FILELOCATION)
        datasetId = requestData.get(pc.DATASETID)
        datasetName = requestData.get(pc.DATASETNAME)
        sentimentCol = requestData.get(pc.PREDICTOR)
        storageLocation = requestData.get(pc.LOCATIONADDRESS)
        modelName = requestData.get(pc.MODELNAME)
        predictionColm = pc.PREDICTION_ + modelName
        sentimentType = requestData.get(pc.SENTIMENTTYPE)
        conversationIdCol = requestData.get(pc.CONVERSATIONIDCOL)
        dateCol = requestData.get(pc.DATECOL)
        isCumulative = requestData.get(pc.ISCUMULATIVE)
        sentimentResultDatasetPath = requestData.get(pc.SENTIMENTRESULTDATASETPATH)
        indexerPathMapping = {}
        encoderPathMapping = {}

        self.predictiveData.update({
            pc.STORAGELOCATION: storageLocation,
            pc.SENTIMENTDATASETPATH: fileLocation,
            pc.SENTIMENTCOLNAME: sentimentCol,
            pc.PREDICTIONCOLM: predictionColm,
            pc.SENTIMENTTYPE: sentimentType,
            pc.ALGORITHMNAME: sentimentType,
            pc.SENTIMENTRESULTDATASETPATH: sentimentResultDatasetPath,
            pc.CONVERSATIONIDCOL: conversationIdCol,
            pc.DATECOL: dateCol,
            pc.ISCUMULATIVE: isCumulative,
            pc.DATASETID: datasetId,
            pc.MODELNAME: modelName,
            pc.DATASETNAME: datasetName,
            pc.INDEXERPATHMAPPING: indexerPathMapping,
            pc.ONEHOTENCODERPATHMAPPING: encoderPathMapping,
        })





    def createFeatureSelectionData(self, requestData):
        fileLocation = requestData.get(pc.FILELOCATION)
        feature_colm_req = requestData.get(pc.PREDICTOR)
        label_colm_req = requestData.get(pc.TARGET)
        algo_name = requestData.get(pc.ALGORITHMNAME)
        relation_list = requestData.get(pc.RELATIONSHIP_LIST)
        relation = requestData.get(pc.RELATIONSHIP)
        featureId = requestData.get(pc.FEATUREID)
        requestType = requestData.get(pc.REQUESTTYPE)
        locationAddress = requestData.get(pc.LOCATIONADDRESS)
        datasetName = requestData.get(pc.DATASETNAME)
        modelSheetName = requestData.get(pc.MODELSHEETNAME)

        self.predictiveData.update({
            pc.DATASETADD: fileLocation,
            pc.FEATURESCOLM: feature_colm_req,
            pc.LABELCOLM: label_colm_req,
            pc.ALGORITHMNAME: algo_name,
            pc.RELATIONSHIP_LIST: relation_list,
            pc.RELATIONSHIP: relation,
            pc.REQUESTTYPE: requestType,
            pc.LOCATIONADDRESS: locationAddress,
            pc.DATASETNAME: datasetName,
            pc.MODELID: featureId,
            pc.MODELSHEETNAME: modelSheetName,
        })
        return self.predictiveData

    def createRegressionModelData(self, requestData):
        fileLocation = requestData.get(pc.FILELOCATION)
        feature_colm_req = requestData.get(pc.PREDICTOR)
        label_colm_req = requestData.get(pc.TARGET)
        algo_name = requestData.get(pc.ALGORITHMNAME)
        relation_list = requestData.get(pc.RELATIONSHIP_LIST)
        relation = requestData.get(pc.RELATIONSHIP)
        trainDataLimit = requestData.get(pc.TRAINDATALIMIT)
        modelId = requestData.get(pc.MODELID)
        requestType = requestData.get(pc.REQUESTTYPE)
        locationAddress = requestData.get(pc.LOCATIONADDRESS)
        datasetName = requestData.get(pc.DATASETNAME)
        modelSheetName = requestData.get(pc.MODELSHEETNAME)
        modelSheetName = pc.PREDICTION_ + modelSheetName

        self.predictiveData.update({
            pc.DATASETADD: fileLocation,
            pc.FEATURESCOLM: feature_colm_req,
            pc.LABELCOLM: label_colm_req,
            pc.ALGORITHMNAME: algo_name,
            pc.RELATIONSHIP_LIST: relation_list,
            pc.RELATIONSHIP: relation,
            pc.TRAINDATALIMIT: trainDataLimit,
            pc.MODELID: modelId,
            pc.REQUESTTYPE: requestType,
            pc.LOCATIONADDRESS: locationAddress,
            pc.DATASETNAME: datasetName,
            pc.MODELSHEETNAME: modelSheetName,
        })
        return self.predictiveData

    def createPredictionData(self, requestData):
        fileLocation = requestData.get(pc.FILELOCATION)
        feature_colm_req = requestData.get(pc.PREDICTOR)
        label_colm_req = requestData.get(pc.TARGET)
        algo_name = requestData.get(pc.ALGORITHMNAME)
        relation_list = requestData.get(pc.RELATIONSHIP_LIST)
        relation = requestData.get(pc.RELATIONSHIP)
        modelId = requestData.get(pc.MODELID)
        requestType = requestData.get(pc.REQUESTTYPE)
        modelStorageLocation = requestData.get(pc.MODELSTORAGELOCATION)
        locationAddress = requestData.get(pc.LOCATIONADDRESS)
        datasetName = requestData.get(pc.DATASETNAME)
        modelSheetName = requestData.get(pc.MODELSHEETNAME)
        modelSheetName = pc.PREDICTION_ + modelSheetName

        self.predictiveData.update({
            pc.DATASETADD: fileLocation,
            pc.FEATURESCOLM: feature_colm_req,
            pc.LABELCOLM: label_colm_req,
            pc.ALGORITHMNAME: algo_name,
            pc.RELATIONSHIP_LIST: relation_list,
            pc.RELATIONSHIP: relation,
            pc.MODELID: modelId,
            pc.REQUESTTYPE: requestType,
            pc.LOCATIONADDRESS: locationAddress,
            pc.DATASETNAME: datasetName,
            pc.MODELSHEETNAME: modelSheetName,
            pc.MODELSTORAGELOCATION: modelStorageLocation
        })
        return self.predictiveData
