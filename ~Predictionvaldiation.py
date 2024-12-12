from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexerModel

# Local paths for the model and validation dataset
model_path = "./TrainedModel"  # Path to the locally saved trained model
validation_data_path = "./ValidationDataset.csv"  # Path to the local validation dataset

# Step 1: Initialize SparkSession
spark = SparkSession.builder \
    .appName("WineQualityValidation") \
    .getOrCreate()

# Step 2: Load Validation Dataset
validation_data = spark.read.csv(validation_data_path, header=True, inferSchema=True, sep=";")

# Clean column names
validation_data = validation_data.toDF(*[col.strip().replace('"', '').replace(' ', '_') for col in validation_data.columns])

# Debug: Print validation dataset columns
print("Validation Dataset Columns:", validation_data.columns)

# Step 3: Load the Trained Model
model = PipelineModel.load(model_path)

# Step 4: Make Predictions
predictions = model.transform(validation_data)

# Step 5: Evaluate F1 Score
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)
print(f"Validation F1 Score: {f1_score:.4f}")

# Debug: Display sample predictions
predictions.select("quality", "prediction", "probability").show(10)

