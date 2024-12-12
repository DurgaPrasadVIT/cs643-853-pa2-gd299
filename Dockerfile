# Step 1: Use an official Python image with Java installed for Spark compatibility
FROM openjdk:11-jre-slim

# Step 2: Install Python and necessary dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Install PySpark, NumPy, and other Python dependencies
RUN pip3 install pyspark pandas numpy

# Step 4: Set up directories for the application
WORKDIR /app

# Step 5: Copy your Spark script, model, and dataset into the container
COPY ./Predictionvaldiation.py ./Predictionvaldiation.py
COPY ./TrainedModel ./TrainedModel
COPY ./ValidationDataset.csv ./ValidationDataset.csv

# Step 6: Set the entrypoint to use spark-submit
ENTRYPOINT ["spark-submit", "Predictionvaldiation.py"]
