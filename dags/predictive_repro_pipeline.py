from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(seconds=30),
    'execution_timeout': timedelta(minutes=30),
    'append_env': True,
}

def log_failure_callback(context):
    run_id = context.get('run_id')
    task_id = context.get('task_instance').task_id
    print(f"ALERT: Task {task_id} failed in DAG Run: {run_id}.")

"""Rationale: schedule=None is used because this pipeline is manually triggered for deterministic reproducibility checks, not time-based scheduling."""
with DAG(
    'predictive_repro_pipeline',
    default_args=default_args,
    description='End-to-End Orchestration for Spotify Ablation Study',
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['dss150p', 'reproducibility'],
    params={
        "ingestion_mode": "full",
        "random_state": 42
    }
) as dag:

    task_env = {
        "PIPELINE_RUN_ID": "{{ run_id }}",
        "PIPELINE_RANDOM_STATE": "{{ params.random_state }}",
    }

    ingest_source = BashOperator(
        task_id='ingest_source',
        bash_command='cd /opt/airflow && python -m src.ingest --mode {{ params.ingestion_mode }}',
        execution_timeout=timedelta(minutes=5),
        on_failure_callback=log_failure_callback,
        env=task_env,
    )

    validate_raw = BashOperator(
        task_id='validate_raw',
        bash_command='cd /opt/airflow && python -m src.validate',
        on_failure_callback=log_failure_callback,
        env=task_env,
    )

    transform_staging = BashOperator(
        task_id='transform_staging',
        bash_command='cd /opt/airflow && python -m src.transform --stage staging',
        on_failure_callback=log_failure_callback,
        env=task_env,
    )

    build_curated = BashOperator(
        task_id='build_curated',
        bash_command='cd /opt/airflow && python -m src.transform --stage curated',
        on_failure_callback=log_failure_callback,
        env=task_env,
    )

    load_postgres = BashOperator(
        task_id='load_postgres',
        bash_command='cd /opt/airflow && python -m src.storage',
        on_failure_callback=log_failure_callback,
        env=task_env,
    )

    validate_curated = BashOperator(
        task_id='validate_curated',
        bash_command='cd /opt/airflow && python -m src.validate --stage curated',
        on_failure_callback=log_failure_callback,
        env=task_env,
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='cd /opt/airflow && python -m src.train',
        on_failure_callback=log_failure_callback,
        env=task_env,
    )

    evaluate_model = BashOperator(
        task_id='evaluate_model',
        bash_command='cd /opt/airflow && python -m src.evaluate',
        on_failure_callback=log_failure_callback,
        env=task_env,
    )

    reproducibility_check = BashOperator(
        task_id='reproducibility_check',
        bash_command='cd /opt/airflow && python -m src.reproducibility_check',
        retries=0,
        on_failure_callback=log_failure_callback,
        env=task_env,
    )

    publish_outputs = BashOperator(
        task_id='publish_outputs',
        bash_command='cd /opt/airflow && python -m src.publish',
        on_failure_callback=log_failure_callback,
        env=task_env,
    )

    ingest_source >> validate_raw >> transform_staging >> build_curated >> load_postgres
    load_postgres >> validate_curated >> train_model >> evaluate_model >> reproducibility_check >> publish_outputs
