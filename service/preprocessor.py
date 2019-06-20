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

class preprocessor:
    def __init__(self,fname):
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
        maldataset = sc.textFile(fname)
        trainHeader = maldataset.first()
        maldataset = maldataset.filter(lambda line: line != trainHeader).mapPartitions(lambda x: csv.reader(x))
        maldataset = maldataset.map(lambda l:self.fillna(l))
        cats = self.set_cat_indx_map(maldataset.collect())
        maldataset = maldataset.map(lambda l:self.replace_cats(l,cats))
        o = ''
        for l in maldataset.collect():
            for i in range(len(l)):
                o += ','.join(l) + '\n'
        ofile = open('dataset.csv','w')
        ofile.write(o)
        ofile.close()
        print(maldataset.first())
    
    def replace_cats(self,r,cats):
        for i in range(len(r)):
            if (i,r[i]) in cats:
                r[i] = str(cats[i,r[i]])
        return r
    
    def set_cat_indx_map(self,ds):
        l = len(ds)
        cats = {}
        for i in range(l):
            for j in range(len(ds[i])):
                v = ds[i][j]
                if (j,v) not in cats:
                    c = sum([ 1 for (idx,c) in cats if idx == j])
                    cats[j,v] = c
                ds[i][j] = cats[j,v]
        return cats
                    
    def fillna(self,line):
        for i in range(len(line)):
            if line[i] == '':
                line[i] = '0'
            line[i] = str(line[i])
        return line   
    
    def toint(self,line):
        for i in range(len(line)):
            if line[i] == '':
                line[i] = '0'
        return line


