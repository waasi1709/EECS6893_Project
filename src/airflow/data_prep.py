from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql.functions import trim, lower
from pyspark.sql.functions import col, when, lit

def prep_data():
    # """ Through this module , our aim is to perform a comprehensive data cleaning, anomaly detection, and preprocessing pipeline for the Medicare dataset. 
    #     The Dataset is obtained from CMS Medicare, which is a national insurance data aggregator for 2022, it contains approx 10 million rows
    #     We have several columns in the data such as provider , patient names , provider ID's etc. """

    #I have initialized a spark session and loaded the data from my GCP bucket into a spark dataframe.

    spark = SparkSession.builder.appName("BigDataProject").getOrCreate()
    data_path = "gs://6893_waj/Medicare_Physician_Other_Practitioners_by_Provider_and_Service_2022.csv"
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    #Lets rename columns for better interpretability.
    renamed_df = df.select(
        col("Rndrng_NPI").alias("Provider_ID"),
        col("Rndrng_Prvdr_Last_Org_Name").alias("Provider_Last_Name"),
        col("Rndrng_Prvdr_First_Name").alias("Provider_First_Name"),
        col("Rndrng_Prvdr_Crdntls").alias("Provider_Credentials"),
        col("Rndrng_Prvdr_Type").alias("Provider_Type"),
        col("Rndrng_Prvdr_State_Abrvtn").alias("Provider_State"),
        col("Rndrng_Prvdr_RUCA").alias("Provider_RUCA_Code"),
        col("Rndrng_Prvdr_RUCA_Desc").alias("Provider_RUCA_Description"),
        col("Rndrng_Prvdr_Mdcr_Prtcptg_Ind").alias("Medicare_Participation"),
        col("HCPCS_Cd").alias("Procedure_Code"),
        col("HCPCS_Desc").alias("Procedure_Description"),
        col("Tot_Benes").alias("Total_Beneficiaries"),
        col("Tot_Srvcs").alias("Total_Services"),
        col("Tot_Bene_Day_Srvcs").alias("Total_Beneficiary_Days"),
        col("Avg_Sbmtd_Chrg").alias("Average_Submitted_Charge"),
        col("Avg_Mdcr_Alowd_Amt").alias("Average_Medicare_Allowed"),
        col("Avg_Mdcr_Pymt_Amt").alias("Average_Medicare_Payment"),
        col("Avg_Mdcr_Stdzd_Amt").alias("Average_Standardized_Payment")
    )

    #Next lets drop rows with no provider_id or Procedure_Code as these are extremely critical and info without this is useless.
    critical_columns = ["Provider_ID", "Procedure_Code"]  
    cleaned_df = renamed_df.dropna(subset=critical_columns)


    numeric_columns = [
        "Total_Beneficiaries",
        "Total_Services",
        "Total_Beneficiary_Days",
        "Average_Submitted_Charge",
        "Average_Medicare_Allowed",
        "Average_Medicare_Payment",
        "Average_Standardized_Payment"
    ]

    #Fill missing numerical columns with the median so that Nan and 0's are avoided.
    #Median is robust against outliers, ensuring quality of the data is maintained.
    #calculate_median computes the median using the approx quantile method present in pyspark , this enables us to modify data at scale i.e. Big Data.
    def calculate_median(df, column):
        return df.approxQuantile(column, [0.5], 0.01)[0]
    for col_name in numeric_columns:
        median_value = calculate_median(cleaned_df, col_name)
        cleaned_df = cleaned_df.withColumn(
            col_name,
            when(col(col_name).isNull(), lit(median_value)).otherwise(col(col_name))
        )

    #We define the categorical columns.
    categorical_columns = [
        "Provider_Last_Name",
        "Provider_First_Name",
        "Provider_Credentials",
        "Provider_Type",
        "Provider_State",
        "Provider_RUCA_Description"
    ]

    #Fill missing Categorical columns with "Unknown", we do not drop the entire row to prevent loss of data.
    for col_name in categorical_columns:
        cleaned_df = cleaned_df.withColumn(
            col_name,
            when(col(col_name).isNull(), lit("Unknown")).otherwise(col(col_name))
        )

    #Remove duplicates and use lowercase letters for uniformity.
    cleaned_df = cleaned_df.dropDuplicates()
    cleaned_df = cleaned_df.withColumn("Provider_Last_Name", trim(lower(col("Provider_Last_Name")))) \
                        .withColumn("Provider_First_Name", trim(lower(col("Provider_First_Name")))) \
                        .withColumn("Provider_Credentials", trim(lower(col("Provider_Credentials")))) \
                        .withColumn("Provider_Type", trim(lower(col("Provider_Type"))))

    #Next lets create a mechanism to detect some anomalies , I have used a mix of the Z-score and IQR approach for this.
    #The core ideas behind the methods are,
    # 1) Z-Score Method: Flags values that deviate significantly (beyond 3 standard deviations) from the mean.
    # 2) IQR Method: Flags values outside 1.5 times the interquartile range (IQR).
    from pyspark.sql.functions import col, when, lit, mean, stddev

    # Define the thresholds
    z_threshold = 3
    iqr_multiplier = 1.5  

    #We calculate the mean and standard deviation for each column in the numeric_columns list, 
    #This is required for the Z-score calculation.
    stats_df = cleaned_df.select(
        *[mean(col(c)).alias(f"{c}_mean") for c in numeric_columns],
        *[stddev(col(c)).alias(f"{c}_stddev") for c in numeric_columns]
    ).collect()[0]

    # Here this code applies two methods, Z-Score and IQR (Interquartile Range), to identify outliers in each numeric column. 
    # Outliers are flagged by creating new columns in the cleaned_df DataFrame, indicating whether each value is an outlier according to the respective method.
    for col_name in numeric_columns:
        #Extract precomputed stats for Z-score
        mean_value = stats_df[f"{col_name}_mean"]
        stddev_value = stats_df[f"{col_name}_stddev"]
        
        #Z-Score method
        cleaned_df = cleaned_df.withColumn(
            f"Z_Outlier_Flag_{col_name}",
            when((col(col_name) > mean_value + z_threshold * stddev_value) |
                (col(col_name) < mean_value - z_threshold * stddev_value), 1).otherwise(0)
        )
        
        #IQR method
        Q1, Q3 = cleaned_df.approxQuantile(col_name, [0.25, 0.75], 0.01)
        IQR = Q3 - Q1
        
        cleaned_df = cleaned_df.withColumn(
            f"IQR_Outlier_Flag_{col_name}",
            when((col(col_name) < Q1 - iqr_multiplier * IQR) | (col(col_name) > Q3 + iqr_multiplier * IQR), 1).otherwise(0)
        )

    #Lets Combine Z-Score and IQR flags for a mixed approach and have a new column Mixed_Outlier_Flag.
    #By applying both methods, this code provides a comprehensive approach to outlier detection, accommodating different data distributions.
    for col_name in numeric_columns:
        cleaned_df = cleaned_df.withColumn(
            f"Mixed_Outlier_Flag_{col_name}",
            when((col(f"Z_Outlier_Flag_{col_name}") == 1) | (col(f"IQR_Outlier_Flag_{col_name}") == 1), 1).otherwise(0)
        )

    # Valid charge ratio is one of the most important aspects of anomaly detection. 
    # The Valid Charge Ratio is used to verify the consistency between two columns:
    # Average_Submitted_Charge: The charge submitted by the provider for a service.
    # Average_Medicare_Allowed: The amount Medicare allows for that service.
    # We can later use this column for anomaly detection with some fixed criteria.
    cleaned_df = cleaned_df.withColumn(
        "Valid_Charge_Ratio",
        col("Average_Submitted_Charge") / col("Average_Medicare_Allowed")
    ).filter(col("Valid_Charge_Ratio") <= 10)

    #Final Dataframe with required outlier flags.
    cleaned_df.select(
        *numeric_columns,
        *[f"Z_Outlier_Flag_{col}" for col in numeric_columns],
        *[f"IQR_Outlier_Flag_{col}" for col in numeric_columns],
        *[f"Mixed_Outlier_Flag_{col}" for col in numeric_columns],
        "Valid_Charge_Ratio"
    ).show(10, truncate=False)


    #Validate Data Consistency
    cleaned_df = cleaned_df.filter((col("Average_Submitted_Charge") >= col("Average_Medicare_Allowed")) &
                                    (col("Average_Submitted_Charge") <= 10 * col("Average_Medicare_Allowed")))

    #Save the Cleaned DataFrame, the reason i have stored it as a parquet and not csv is to preserve the data types and pyspark sessions can easily read
    # parquet files.
    output_path = "gs://6893_waj/cleaned_data.parquet"
    cleaned_df.write.parquet(output_path, mode="overwrite")

    #Some verification and debugging steps to ensure our data is clean and as required.
    print("Cleaned DataFrame Preview:")
    cleaned_df.show(5, truncate=False)
    return