from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

# กำหนดชื่อและเวลาเริ่ม
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'catchup': False,
}

with DAG('my_data_science_pipeline', default_args=default_args, schedule_interval=None) as dag:

    # Task 1: Clean Data
    t1_clean = BashOperator(
        task_id='clean_data',
        bash_command='python /opt/airflow/scripts/clean_data.py'
    )

    # Task 2: Scrape Data
    t2_scrape = BashOperator(
        task_id='scrape_data',
        bash_command='python /opt/airflow/scripts/web_scrape.py'
    )

    # Task 3: Prediction
    t3_predict = BashOperator(
        task_id='prediction',
        bash_command='python /opt/airflow/scripts/prediction.py'
    )

    # Task 4: Clustering
    t4_cluster = BashOperator(
        task_id='clustering',
        bash_command='python /opt/airflow/scripts/clustering.py'
    )

    # กำหนดลำดับการทำงาน (Dependency)
    # t1 ทำเสร็จก่อน -> แล้ว t2, t3, t4 ทำพร้อมกันได้เลย
    t1_clean >> [t2_scrape, t3_predict, t4_cluster]