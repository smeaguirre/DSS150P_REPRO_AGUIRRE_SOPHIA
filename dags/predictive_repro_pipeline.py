"""
DAG for Spotify Track Popularity Reproducibility Pipeline
Rationale for schedule=None: This is an analytical reproducibility pipeline meant to be triggered 
manually when testing against new baseline models or performing an isolated reproduction audit, 
rather than a continuous daily ingestion feed.
"""
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

# Default settings applied to all tasks
default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2, # Retries for tasks where retry is safe (Acceptance Test A9 ensures DB is safe)
    'retry_delay': timedelta(seconds=30),
}

def log_failure_callback(context):
    """Failure handling that logs useful context"""
    run_id = context.get('run_id')
    task_id = context.get('task_instance').task_id
    print(f"ALERT: Task {task_id} failed in DAG Run: {run_id}.")

with DAG(
    'predictive_repro_pipeline',
    default_args=default_args,
    description='End-to-End Orchestration for Spotify Ablation Study',
    schedule_interval=None, # Documented manual-trigger rationale
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['dss150p', 'reproducibility'],
    params={
        "ingestion_mode": "full", # Runtime parameter
        "random_state": 42
    }
) as dag:

    # 1. Ingest
    ingest_source = BashOperator(
        task_id='ingest_source',
        bash_command='cd /opt/airflow && python src/ingest.py --mode {{ params.ingestion_mode }}',
        execution_timeout=timedelta(minutes=5), # Execution timeout control
        on_failure_callback=log_failure_callback,
        env={"PIPELINE_RUN_ID": "{{ run_id }}"} # Consistent run identifier
    )

    # 2. Validate Raw 
    validate_raw = BashOperator(
        task_id='validate_raw',
        bash_command='cd /opt/airflow && python src/validate.py',
        on_failure_callback=log_failure_callback,
        env={"PIPELINE_RUN_ID": "{{ run_id }}"}
    )

    # 3 & 4. Transform Staging and Build Curated
    transform_staging = BashOperator(
        task_id='transform_staging',
        bash_command='cd /opt/airflow && python src/transform.py',
        on_failure_callback=log_failure_callback,
        env={"PIPELINE_RUN_ID": "{{ run_id }}"}
    )
    
    build_curated = EmptyOperator(task_id='build_curated')

    # 5. Load PostgreSQL
    load_postgres = BashOperator(
        task_id='load_postgres',
        bash_command='cd /opt/airflow && python src/storage.py',
        on_failure_callback=log_failure_callback,
        env={
            "PIPELINE_RUN_ID": "{{ run_id }}",
            "POSTGRES_HOST": "postgres",
            "POSTGRES_USER": "airflow",
            "POSTGRES_PASSWORD": "airflow",
            "POSTGRES_DB": "airflow"
        }
    )

    # 6. Validate Curated
    validate_curated = EmptyOperator(task_id='validate_curated')

    # 7. Train Model
    train_model = BashOperator(
        task_id='train_model',
        bash_command='cd /opt/airflow && python src/train.py',
        on_failure_callback=log_failure_callback,
        env={
            "PIPELINE_RUN_ID": "{{ run_id }}",
            "POSTGRES_HOST": "postgres",
            "POSTGRES_USER": "airflow",
            "POSTGRES_PASSWORD": "airflow",
            "POSTGRES_DB": "airflow"
        }
    )

    # 8. Evaluate Model
    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command='cd /opt/airflow && python src/evaluate.py',
        on_failure_callback=log_failure_callback,
        env={
            "PIPELINE_RUN_ID": "{{ run_id }}",
            "POSTGRES_HOST": "postgres",
            "POSTGRES_USER": "airflow",
            "POSTGRES_PASSWORD": "airflow",
            "POSTGRES_DB": "airflow"
        }
    )
    
    # 9. Reproducibility Check (Placeholder for Milestone 09)
    reproducibility_check = EmptyOperator(task_id='reproducibility_check')

    # 10. Publish Outputs
    publish_outputs = EmptyOperator(task_id='publish_outputs')

    # Define Explicit Dependencies
    ingest_source >> validate_raw >> transform_staging >> build_curated >> load_postgres
    load_postgres >> validate_curated >> train_model >> evaluate_model >> reproducibility_check >> publish_outputs