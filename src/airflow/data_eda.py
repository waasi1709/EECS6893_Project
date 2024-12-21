from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum


def clean_data():
    # """ We will do some Exploratory Data Analysis(EDA) on this preprocessed data in this module. """

    spark = SparkSession.builder.appName("ParquetDataAnalysis").getOrCreate()

    #Load cleaned data
    input_path = "gs://6893_waj/cleaned_data.parquet"
    cleaned_df = spark.read.parquet(input_path)

    #Check the schema of the preprocessed data for consistency and to ensure all columns required for anomaly detection criteria are readily available.
    #We check several other metrics as well to understand the data and its distribution better.
    cleaned_df.printSchema()
    cleaned_df.show(20, truncate=False)  
    record_count = cleaned_df.count()
    print(f"Total number of records: {record_count}")
    cleaned_df.describe().show()
    numeric_columns = [
        "Total_Beneficiaries", "Total_Services", "Total_Beneficiary_Days",
        "Average_Submitted_Charge", "Average_Medicare_Allowed", "Average_Medicare_Payment"
    ]
    cleaned_df.select(*numeric_columns).describe().show()

    #A DataFrame showing the count of missing values for every column.
    missing_values = cleaned_df.select(
        *[sum(col(c).isNull().cast("int")).alias(c) for c in cleaned_df.columns]
    )
    missing_values.show()

    #A summary table showing each Provider_Type and the corresponding average submitted charge.
    cleaned_df.groupBy("Provider_Type").agg(
        {"Average_Submitted_Charge": "avg"}
    ).show()

    #A summary table showing the number of providers in each state.
    cleaned_df.groupBy("Provider_State").count().show()

    #A filtered DataFrame showing all rows where the Average_Submitted_Charge is greater than 1000.
    high_charges = cleaned_df.filter(col("Average_Submitted_Charge") > 1000)
    high_charges.show()
    return
