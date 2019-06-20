import time
import pyspark
import os
import csv
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import Row
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from preprocessor import preprocessor
import os

class mlib_classifier:
    def __init__(self):
        if not os.path.exists('dataset.csv'):
            preprocessor('kaggle.csv')

        
    def GetFScore(self,i,ratio):
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
        maldataset = sc.textFile("dataset.csv")
        trainHeader = maldataset.first()
        maldataset = maldataset.filter(lambda line: line != trainHeader).mapPartitions(lambda x: csv.reader(x))
        maldataset = maldataset.map(lambda l:self.toint(l))
        df = maldataset.map(lambda l:(l[-1],Vectors.dense(l[0:-1])))
        maldataset = maldataset.map(lambda line: LabeledPoint(line[-1],[line[0:len(line)-1]]))
        trainData, testData = maldataset.randomSplit([ratio,1 - ratio])
        if i > 0:
            return self.BC(trainData, testData,i)

        df = spark.createDataFrame(df.collect(), ["label", "features"])
        splits = df.randomSplit([ratio,1- ratio], 1234)
        train = splits[0]
        test = splits[1]
        mlp = MultilayerPerceptronClassifier(maxIter=100, layers=[35, 100, 100], blockSize=1, seed=123)
        model = mlp.fit(train)
        result = model.transform(test)
        predictionAndLabels = result.select("prediction", "label")
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        return evaluator.evaluate(predictionAndLabels)
        
    def BC(self,trainData, testData,i):
        model = None
        if i == 1:
            model = RandomForest.trainClassifier(trainData, numClasses = 2,categoricalFeaturesInfo = {}, numTrees = 100,
                    featureSubsetStrategy='auto', impurity='gini', maxDepth=12,
                    maxBins=32, seed=None)
            return self.fscore(model,testData)
        if i == 2:
            model = DecisionTree.trainClassifier(trainData, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=12, maxBins=32)
            return self.fscore(model,testData)
            

        
        
        
    def fscore(self,model,testData):
        predictions = model.predict(testData.map(lambda x: x.features))
        labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
        metrics = BinaryClassificationMetrics(labelsAndPredictions)
        return metrics.areaUnderPR
        

        
    def toint(self,line):
        for i in range(len(line)):
            if line[i] == '':
                line[i] = '0'
            line[i] = int(line[i])
        return line



cf = mlib_classifier()
score = cf.GetFScore(1,.2)