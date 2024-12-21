from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


def detect_outliers():
    # """The objective of this module is to create a column is_fraud for our dataset based on the Anomaly flags we have built earlier 
    #    which includes Z-score based outlier flags , IQR based outlier flags and mixed_outlier flag for certain columns , the valid charge ratio also 
    #    remains extremely important in this task ."""

    #Initialize spark and load dataset
    spark = SparkSession.builder.appName("RefineFraudCriteria").getOrCreate()
    input_path = "gs://6893_waj/cleaned_data.parquet"
    cleaned_df = spark.read.parquet(input_path)

    #This code defines a fraud detection criteria by combining several conditions that may indicate potential anomalies or fraud. 
    #Firstly we check if specific columns (Average_Submitted_Charge, Average_Medicare_Allowed, and Average_Medicare_Payment) are flagged as 
    #outliers based on the mixed outlier detection method (Z-Score or IQR).
    #If any of these outlier flags are set to 1, it indicates that the corresponding value deviates significantly from expected norms, 
    # suggesting possible anomalies.

    #Secondly , A Valid_Charge_Ratio greater than 10 implies that the submitted charge is more than 10 times the Medicare-allowed amount, 
    #which is unusually high and may indicate suspicious billing.
    refined_fraud_criteria = (
        (col("Mixed_Outlier_Flag_Average_Submitted_Charge") == 1) |
        (col("Mixed_Outlier_Flag_Average_Medicare_Allowed") == 1) |
        (col("Mixed_Outlier_Flag_Average_Medicare_Payment") == 1) |
        (col("Valid_Charge_Ratio") > 10)
    )

    #The is_fraud column is created basis the fraud criteria we set above
    cleaned_df = cleaned_df.withColumn("is_fraud", when(refined_fraud_criteria, 1).otherwise(0))

    #The distribution of the is_fraud column is noted 
    fraud_distribution = cleaned_df.groupBy("is_fraud").count()
    fraud_distribution.show()

    #Upload the saved and final dataframe to GCP Bucket
    output_path = "gs://6893_waj/FraudLabelledData.parquet"
    cleaned_df.write.parquet(output_path, mode="overwrite")
    print("Updated DataFrame with fraud labels saved.")

    return
