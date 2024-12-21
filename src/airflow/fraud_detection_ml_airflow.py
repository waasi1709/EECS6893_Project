from datetime import datetime, timedelta
from textwrap import dedent
import time
import data_prep,data_eda,data_outlier_detection, data_labelling, train_logistic_regression, train_random_forest, report_collection


from airflow import DAG

from airflow.operators.python import PythonOperator


def collect_data():
   data_prep.prep_data()
   return


def preprocess_data():
    data_eda.clean_data
    return

def detect_outlier():
   data_outlier_detection.detect_outliers()
   return

def label_data():
   data_labelling.encode_data()
   return
   
def train_lr_model():
    train_logistic_regression.train_lr()
    return

def train_rf_model():
   train_random_forest.train_rf()
   return()

def collect_report():
    input_csv_files = ['lr_metrics.csv','rf_metrics.csv']
    output_file = 'consolidated_metrics.csv'

    report_collection.consolidate_csv_files(input_csv_files,output_file)
    return



############################################
# DEFINE AIRFLOW DAG (SETTINGS + SCHEDULE)
############################################

default_args = {
    'owner': 'gautamvr',
    'depends_on_past': False,
    'email': ['gv2359@columbia.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=30),
    'provide_context': True,
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

with DAG(
    'fraud_detection_ml_airflow',
    default_args=default_args,
    description='Fraud Detection pipeline for Medicare Fraud',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 10, 17),
    catchup=False,
    tags=['example'],
) as dag:


    #Tasks for APPLE stocks

    Fraud_DataCollection = PythonOperator(
        task_id='Fraud_DataCollection',
        python_callable=collect_data,
        retries=3,
        op_args = ['source_path']
    )

    FraudDataPreProcessing = PythonOperator(
        task_id='FraudDataPreProcessing',
        python_callable=preprocess_data,
        retries=3,
        op_args = ['features']
    )

    OutlierDetection = PythonOperator(
        task_id='OutlierDetection',
        python_callable=detect_outlier,
        retries=3,
        op_args = ['outlier']
    )

    FraudLabelling = PythonOperator(
        task_id='FraudLabelling',
        python_callable=label_data,
        retries=3,
        op_args = ['labels']
    )

    LR_ModelTrainingPredict = PythonOperator(
        task_id='LR_ModelTrainingPredict',
        python_callable=train_lr_model,
        retries=3,
        op_args = ['accuracy']
    )

    RF_ModelTrainingPredict = PythonOperator(
        task_id='RF_ModelTrainingPredict',
        python_callable=train_rf_model,
        retries=3,
        op_args = ['accuracy']
    )

    ReportCollection = PythonOperator(
        task_id='ReportCollection',
        python_callable=collect_report,
        retries=3
    )

##########################################
# DEFINE TASKS HIERARCHY
##########################################

    # task dependencies 

    Fraud_DataCollection >> FraudDataPreProcessing >> OutlierDetection >> FraudLabelling >> [LR_ModelTrainingPredict, RF_ModelTrainingPredict] 
    [LR_ModelTrainingPredict, RF_ModelTrainingPredict] >> ReportCollection

    

