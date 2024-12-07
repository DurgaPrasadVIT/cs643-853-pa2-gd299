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

