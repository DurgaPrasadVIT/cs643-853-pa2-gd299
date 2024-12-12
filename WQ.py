from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Paths to S3 datasets and model save location
training_data_path = "s3a://aws-logs-712835584163-us-east-1/elasticmapreduce/WineQuality/TrainingDataset.csv"
validation_data_path = "s3a://aws-logs-712835584163-us-east-1/elasticmapreduce/WineQuality/ValidationDataset.csv"
model_save_path = "s3a://aws-logs-712835584163-us-east-1/elasticmapreduce/WineQuality/TrainedModel"

# Step 1: Initialize SparkSession with S3 configurations
spark = SparkSession.builder \
    .appName("WineQualityTraining") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
    .getOrCreate()

# Step 2: Load Training Data from S3
training_data = spark.read.csv(training_data_path, header=True, inferSchema=True, sep=";")

# Step 3: Clean Column Names
training_data = training_data.toDF(*[col.strip().replace('"', '').replace(' ', '_') for col in training_data.columns])

# Debug: Print cleaned column names
print("Training Data Columns:", training_data.columns)

# Step 4: Define Features and Label
feature_columns = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
    "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
label_indexer = StringIndexer(inputCol="quality", outputCol="label")

# Step 5: Define Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50, regParam=0.1, elasticNetParam=0.8)

# Step 6: Create Pipeline
pipeline = Pipeline(stages=[assembler, label_indexer, lr])

# Step 7: Train the Model
model = pipeline.fit(training_data)

# Step 8: Save the Trained Model to S3
model.write().overwrite().save(model_save_path)
print(f"Model saved to S3 at: {model_save_path}")

# Step 9: Load Validation Data from S3
validation_data = spark.read.csv(validation_data_path, header=True, inferSchema=True, sep=";")

# Clean column names for validation data
validation_data = validation_data.toDF(*[col.strip().replace('"', '').replace(' ', '_') for col in validation_data.columns])

# Drop 'label' column if it exists
if 'label' in validation_data.columns:
    validation_data = validation_data.drop('label')

# Debug: Print validation dataset columns
print("Validation Dataset Columns:", validation_data.columns)

# Step 10: Make Predictions
predictions = model.transform(validation_data)

# Step 11: Evaluate the Model
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"Validation F1 Score: {f1_score:.4f}")

# Debug: Display some predictions
predictions.select("quality", "prediction", "probability").show(10)

# Stop SparkSession
spark.stop()

