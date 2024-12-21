from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql.functions import col

def encode_data():
    # """ The goal of this module is build and train Machine Learning models for Fraud Detection """

    spark = SparkSession.builder.appName("FraudDetectionPreprocessing").getOrCreate()
    input_path = "gs://6893_waj/FraudLabelledData.parquet"
    df = spark.read.parquet(input_path)

    #Define Categorical and Numerical Columns.
    categorical_columns = ["Provider_Credentials", "Provider_Type", "Provider_State", "Medicare_Participation"]
    numerical_columns = [
        "Total_Beneficiaries", "Total_Services", "Total_Beneficiary_Days",
        "Average_Submitted_Charge", "Average_Medicare_Allowed",
        "Average_Medicare_Payment", "Average_Standardized_Payment"
    ]
    outlier_columns = [col for col in df.columns if col.startswith("Mixed_Outlier_Flag_")]
    target_column = "is_fraud"

    #Encoding categorical features
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_Index").setHandleInvalid("keep")
        for col in categorical_columns
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{col}_Index", outputCol=f"{col}_OHE")
        for col in categorical_columns
    ]

    #Feature Assemler
    feature_columns = numerical_columns + outlier_columns + [f"{col}_OHE" for col in categorical_columns]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    #Scaling Numerical Features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

    #Target Variable
    df = df.withColumn("label", col(target_column).cast("double"))

    #Build the Transformation Pipeline
    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

    transformed_df = pipeline.fit(df).transform(df)
    final_dataset = transformed_df.select("scaled_features", "label")

    #Save Processed Data to a Parquet File
    output_path = "gs://6893_waj/ProcessedFraudData.parquet"
    final_dataset.write.parquet(output_path, mode="overwrite")

    #Review the final dataset
    final_dataset.show(10, truncate=False)
    print(f"Processed data saved to {output_path}")

    return