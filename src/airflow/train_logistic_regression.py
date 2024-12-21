# Split the data into training and testing sets
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
import pandas as pd
import os

def train_lr():
    #Initialize spark and load dataset
    spark = SparkSession.builder.appName("RefineFraudCriteria").getOrCreate()
    input_path = "gs://6893_waj/ProcessedFraudData.parquet"
    final_dataset = spark.read.parquet(input_path)

    (trainingData, testData) = final_dataset.randomSplit([0.7, 0.3])

    # Create a LogisticRegression model
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")

    # Fit the model on the training data
    lrModel = lr.fit(trainingData)

    # Make predictions on the test data
    predictions = lrModel.transform(testData)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    accuracy = evaluator.evaluate(predictions)

    print("Accuracy:", accuracy)

    # Convert to a DataFrame
    perf_df = pd.DataFrame([{"Accuracy": accuracy}])
    output_csv = "lr_metrics.csv"
    # Append to the CSV file if it exists, otherwise create a new file
    if os.path.exists(output_csv):
        perf_df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        perf_df.to_csv(output_csv, mode="w", header=True, index=False)

    return