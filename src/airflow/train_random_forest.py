# Split the data into training and testing sets
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
import pandas as pd
import os

def train_rf():
    #Initialize spark and load dataset
    spark = SparkSession.builder.appName("RefineFraudCriteria").getOrCreate()
    input_path = "gs://6893_waj/ProcessedFraudData.parquet"
    final_dataset = spark.read.parquet(input_path)

    (trainingData, testData) = final_dataset.randomSplit([0.7, 0.3])

    rf = RandomForestClassifier(numTrees=100, maxDepth=5, labelCol="label", featuresCol="scaled_features")

    # Fit the model on the training data
    rfModel = rf.fit(trainingData)

    # Make predictions on the test data
    predictions = rfModel.transform(testData)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    accuracy = evaluator.evaluate(predictions)

    print("Accuracy:", accuracy)

    # Convert to a DataFrame
    perf_df = pd.DataFrame([{"Accuracy": accuracy}])
    output_csv = "rf_metrics.csv"
    # Append to the CSV file if it exists, otherwise create a new file
    if os.path.exists(output_csv):
        perf_df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        perf_df.to_csv(output_csv, mode="w", header=True, index=False)

    return   