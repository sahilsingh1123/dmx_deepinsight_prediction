from OptimusSpark import DIOptimus
from pandas.compat import reduce

def colrename(x):
    reps = ('.', '_'), (' ', '_'), ('(', ''), (')', ''), ('{', ''), ('}', ''), ('\\n', ''), ('\n', ''), ('\\t', ''), ('\t', ''), ('=', '')
    return reduce(lambda a,kv : a.replace(*kv),reps,x)

def convertColumnParquetFormat(df):
    columns = df.columns
    for col in columns:
        if col.strip().__contains__(' '):
            newcol = col.strip().replace(' ', '_')
            newcol = colrename(newcol)
            df = df.cols.rename(col, newcol)
    return df


class CreateFileUtil:
    def __init__(self, requestData):
        self.manager = DIOptimus.getmanager()
        self.filePath = requestData.get("filePath")
        self.haddopPath = requestData.get("hdfsLocation")
        self.fileName = requestData.get("fileName")
        self.requestSubType = requestData.get("requestSubType")
        self.sheetName = requestData.get("sheetName")

    def convertToParquet(self):
        df = None
        try:
            if self.requestSubType == "createCSVToParquet":
                df = self.manager.optimus.load.csv(self.filePath, infer_schema=False)
            elif self.requestSubType == "createExcelToParquet":
                df = self.manager.optimus.load.excel(self.filePath, self.sheetName, infer_schema=False)

            if df is not None:
                df = convertColumnParquetFormat(df)
                df.write.mode("overwrite").parquet(self.haddopPath + self.fileName + '.parquet')
                return "success"
            return "failed"
        except Exception as e:
            print(e)
            return "failed"
