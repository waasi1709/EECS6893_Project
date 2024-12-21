import pandas as pd
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

def consolidate_csv_files(input_csv_files, consolidated_csv):
    """
    Consolidates multiple CSV files into one.

    Args:
        input_csv_files (list of str): List of input CSV file paths.
        consolidated_csv (str): Path to the consolidated output CSV file.
    """
    spark = SparkSession.builder.ap
    df_list = []
    for csv_file in input_csv_files:
        if os.path.exists(csv_file):
            df = spark.read.option("header", "true").csv(csv_file)
            df = df.withColumn("Source", lit(csv_file))  # Add source column for reference
            df_list.append(df)

    if df_list:
        consolidated_df = df_list[0]
        for df in df_list[1:]:
            consolidated_df = consolidated_df.union(df)

        consolidated_df.write.mode("overwrite").option("header", "true").csv(consolidated_csv)

    spark.stop()
