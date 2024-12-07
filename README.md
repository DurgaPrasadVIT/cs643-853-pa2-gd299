# Wine_Quality_PA2
# Wine Quality Prediction using on AWS

## Project Overview
This project centers on creating an application that does a parallel machine learning to predict wine quality with the help of Apache Spark on Amazon AWS. The goal is to spread the training of a predictive model across numerous EC2 instances, fine-tune it, and then host it with Docker for the prediction.

## Dataset Description
The project utilizes the following datasets for model training and evaluation:
- **TrainingDataset.csv**: For training the machine learning model the four EC2 instances are invoked in parallel.
- **ValidationDataset.csv**: Application for cross-validation, hyperparameters tuning.
- **TestDataset.csv**: Can be used for testing the prediction accuracy, if available otherwise the validation dataset can be employed.

## Objective
The primary goal of this project is to build a robust machine learning model for predicting wine quality. The model is trained in parallel on AWS infrastructure, and the prediction performance is evaluated using the F1 score.

## Technologies and Tools
- **Python**: The programming language used for writing the data processing and machine learning scripts.
- **Spark MLlib**: Spark's machine learning library used for model training and evaluation.
- **Amazon EC2**: Cloud infrastructure for running Spark jobs and distributing the workload across multiple instances.
- **Apache Spark**: A distributed computing framework used to train and deploy the model.
- **Docker**: A platform used to containerize the trained model for easy deployment.

### 1. **Cluster Setup on AWS**

#### Launching the EMR Cluster
1. Access the AWS Management Console.
2. Navigate to **EMR (Elastic MapReduce)** and click on **Create Cluster**.
3. Configure the cluster with the following:
   - **Applications**: Choose **Spark** and **Hadoop**.
   - **Instance Types**: Select **m5.xlarge** or similar for both master and worker nodes.
   - **Number of Instances**: Set **1 master** and **4 worker nodes**.
   - **Key Pair**: Use an existing key pair or create a new one for SSH access.
4. Click **Create Cluster** and wait for it to initialize.

#### SSH into the Master Node
- After the cluster is up and running, find the **Master Node DNS** in the EMR console.
- SSH into the master node using the terminal:
```bash
ssh -i your-key.pem hadoop@<MasterNodeDNS>
```

### 2. **Verify Spark Installation**
- Ensure that **PySpark** is installed and working by typing:
```bash
pyspark
```
- If the Spark shell starts successfully, type `exit()` to close the session.

### 3. **Set Up Environment Variables**
- Confirm the paths for Spark and Hadoop are set up correctly by checking the `~/.bashrc` file:
```bash
cat ~/.bashrc
```
- If necessary, add the following to the `~/.bashrc`:
```bash
echo "export SPARK_HOME=/usr/lib/spark" 
echo "export PATH=\$SPARK_HOME/bin:\$PATH" 
echo "export HADOOP_HOME=/usr/lib/hadoop" 
echo "export PATH=\$HADOOP_HOME/bin:\$PATH" 
source ~/.bashrc
```
### 4. **Validate Cluster Configuration**
- Check the cluster's node configuration by running:
```bash
yarn node -list
```
- Ensure that Spark is utilizing all available nodes. Also, verify available resources using:
```bash
spark-submit --status
```

### 5. **Upload Datasets to S3**
- Go to the **S3** section in the AWS console and upload the **TrainingDataset.csv** and **ValidationDataset.csv** files.
- Alternatively, use the AWS CLI to upload:
```bash
aws s3 cp TrainingDataset.csv s3://your-bucket-name/
aws s3 cp ValidationDataset.csv s3://your-bucket-name/
```
- To verify access, run:
```bash
aws s3 ls s3://your-bucket-name/
```

### 6. **Train the Model on EC2**
- SSH into the master node, and run the training script:
```bash
spark-submit WineQualityPrediction.py
```
- Ensure that the model is saved correctly after training.

---

## Docker Setup for Model Deployment

### 1. **Create the Dockerfile**
- Create a `Dockerfile` in the project directory to set up the environment for the model prediction:
```dockerfile
FROM bitnami/spark:latest

# Set working directory
WORKDIR /app

# Ensure root privileges to install dependencies
USER root

# Fix apt-get issues and install dependencies
RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y python3-pip curl wget && \
    pip3 install numpy pandas

# Create Hadoop configuration directory if not exists
RUN mkdir -p /opt/bitnami/hadoop/etc/hadoop

# Add Hadoop configuration for S3 access
RUN echo "fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" >> /opt/bitnami/hadoop/etc/hadoop/core-site.xml && \
    echo "fs.s3a.aws.credentials.provider=com.amazonaws.auth.DefaultAWSCredentialsProviderChain" >> /opt/bitnami/hadoop/etc/hadoop/core-site.xml && \
    echo "fs.s3a.path.style.access=true" >> /opt/bitnami/hadoop/etc/hadoop/core-site.xml

# Download the Hadoop AWS jar and AWS SDK jar (correct URLs)
RUN mkdir -p /opt/bitnami/spark/jars && \
    wget -O /opt/bitnami/spark/jars/hadoop-aws-3.3.1.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.1/hadoop-aws-3.3.1.jar && \
    wget -O /opt/bitnami/spark/jars/aws-java-sdk-bundle-1.11.1026.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar

# Copy the WineQualityPrediction script to the container
COPY WineQualityPrediction.py /app/WineQualityPrediction.py

# Set the entrypoint for the container
ENTRYPOINT ["spark-submit", "/app/WineQualityPrediction.py"]
```

### 2. **Build the Docker Image**
- Build the Docker image using the following command:
```bash
docker build -t wine-quality-prediction 
```

### 3. **Run the Docker Container**
- After building the Docker image, run the container:
```bash
docker run -it wine-quality-prediction
```

## Conclusion
This project demonstrates how to leverage AWS EC2 instances and Apache Spark to perform parallel machine learning tasks. By training the model in parallel on multiple nodes, you can efficiently handle large datasets. After training, the model is deployed in a Docker container, ensuring easy scalability and deployment for prediction tasks. The use of Spark's distributed computing capabilities on AWS maximizes the processing power for training and prediction tasks.

