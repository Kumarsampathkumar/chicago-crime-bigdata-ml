from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, dayofweek, to_timestamp
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("CrimePrediction").getOrCreate()

data = spark.read.csv("chicago_crime.csv", header=True, inferSchema=True)

data = data.dropna()

data = data.withColumn("Date", to_timestamp("Date"))
data = data.withColumn("Hour", hour("Date"))
data = data.withColumn("DayOfWeek", dayofweek("Date"))

indexer = StringIndexer(inputCol="Primary Type", outputCol="label")
data = indexer.fit(data).transform(data)

assembler = VectorAssembler(
    inputCols=["District","Ward","Community Area","Hour","DayOfWeek"],
    outputCol="features"
)

data = assembler.transform(data)

train, test = data.randomSplit([0.8,0.2])

rf = RandomForestClassifier(featuresCol="features", labelCol="label")

model = rf.fit(train)

predictions = model.transform(test)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print("Model Accuracy:", accuracy)
