#""" The main goal of this file is to generate a sample dataset of the data we have labelled through Anomaly detection techniques used earlier.
#    This will help us to study relevant columns and further prepare the data for Supervised Learning models since we now have labelling 
#    obtained through Anomaly detection. """
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("SaveSampleAsCSV").getOrCreate()

#FraudLabelledData is our labelled data , the criteria for labelling is specified in FraudLabelling.py
input_path = "gs://6893_waj/FraudLabelledData.parquet"
df = spark.read.parquet(input_path)

#Sample 50 rows
sample_df = df.limit(50)
output_path = "gs://6893_waj/sample_data.csv"

#Save the sample database as a CSV file for review to the GCP bucket
sample_df.write.option("header", "true").csv(output_path, mode="overwrite")
print(f"Sample of 50 rows saved to {output_path}")
