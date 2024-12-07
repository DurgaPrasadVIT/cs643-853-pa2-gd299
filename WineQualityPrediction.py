from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Paths to S3 bucket datasets and model save location
training_data_path = "s3://aws-logs-712835584163-us-east-1/elasticmapreduce/WineQuality/TrainingDataset.csv"
validation_data_path = "s3://aws-logs-712835584163-us-east-1/elasticmapreduce/WineQuality/ValidationDataset.csv"
model_save_path = "s3://aws-logs-712835584163-us-east-1/elasticmapreduce/WineQuality/TrainedModel"

# Ensure SparkSession is available
spark = SparkSession.getActiveSession()
if spark is None:
    spark = SparkSession.builder.getOrCreate()

# Load training and validation data with the correct delimiter
training_data = spark.read.csv(training_data_path, header=True, inferSchema=True, sep=";")
validation_data = spark.read.csv(validation_data_path, header=True, inferSchema=True, sep=";")

# Print original column names for debugging
print("Original Training Data Columns:", training_data.columns)

# Clean column names to remove extra quotes and spaces
training_data = training_data.toDF(*[col.strip().replace('"', '').replace(' ', '_') for col in training_data.columns])
validation_data = validation_data.toDF(*[col.strip().replace('"', '').replace(' ', '_') for col in validation_data.columns])

# Print cleaned column names for debugging
print("Cleaned Training Data Columns:", training_data.columns)

# Updated feature column names
feature_columns = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
    "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
label_indexer = StringIndexer(inputCol="quality", outputCol="label")

# Define Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50, regParam=0.1, elasticNetParam=0.8)

# Create pipeline
pipeline = Pipeline(stages=[assembler, label_indexer, lr])

# Train the model
model = pipeline.fit(training_data)

# Save the trained model to S3
model.write().overwrite().save(model_save_path)

# Validate the model
predictions = model.transform(validation_data)

# Evaluate F1 score
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"Validation F1 Score: {f1_score}")

