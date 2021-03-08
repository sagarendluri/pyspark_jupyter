import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession \
 .builder \
 .appName("Python Spark SQL basic example") \
 .config("spark.some.config.option", "some-value") \
 .getOrCreate()
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
df1 = spark.read.csv(r"C:\Users\sagar\Downloads\common_samples.csv",header  = True)#common_samples top_5k_319_samples
df2 = spark.read.csv(r"C:\Users\sagar\Downloads\top_5k_319_samples.csv",header  = True)
dff = df1.select('Line','1st Layer Clusters')
from pyspark.sql.functions import *
# df1 = df1.alias('df1')
# df2 = df2.alias('df2')
df = df2.join(dff, df2.index == dff.Line)#.select('df1.new_data')
print(df.count(),len(df.columns))
dfd = df.drop('_c0','index','AC','GT','GC','PV','TV','SB','PsT','LO','Line')
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(dfd) for column in ["1st Layer Clusters"]]
pipeline = Pipeline(stages=indexers)
ddf = pipeline.fit(dfd).transform(dfd)
ddf = ddf.drop("1st Layer Clusters")#,"Name","Ticket","Cabin","Embarked","Sex","Initial")
# titanic_df.printSchema()
# from pyspark.sql.types import IntegerType
final = ddf.select([col(c).cast('int') for c in ddf.columns])
# final.printSchema()
feature = VectorAssembler(inputCols=final.columns[1:],outputCol="features")
feature_vector= feature.transform(final)
(trainingData, testData) = feature_vector.randomSplit([0.8, 0.2],seed = 11)
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="1st Layer Clusters_index", featuresCol="features")
dt_model = dt.fit(trainingData)
dt_prediction = dt_model.transform(testData)
dt_prediction.select("prediction", "1st Layer Clusters_index", "features").show()

# trainingData