import yaml
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

from create_taskgroups import (create_api_taskgroup,
                               create_sql_taskgroup,
                               create_ml_taskgroup)

from general_functions import (plot_data,
                               plot_predictions,
                               post_model_results)



# import config file
with open('config_file.yaml', 'r') as f:
    config = yaml.safe_load(f)

# some default arguments
default_args = {
    'owner': config['app_owner'],
    'depends_on_past': False,
    'retries': 0,
}
# define the DAG
with DAG(
    config['app_name'],
    default_args=default_args,
    description=config['app_description'],
    schedule=None,  # Manual trigger
    start_date=datetime(2022, 1, 1),
    catchup=False,
    tags=[config['app_tag']]
) as dag:
    # dummy start task
    start_task = EmptyOperator(
        task_id='start_task'
    )
    # define the taskgroups
    api_taskgroup = create_api_taskgroup("api_taskgroup")
    sql_taskgroup = create_sql_taskgroup("sql_taskgroup")
    ml_taskgroup = create_ml_taskgroup("ml_taskgroup")
    # data plot task
    plot_data_task = PythonOperator(
        task_id='plot_data_task',
        python_callable=plot_data
    )
    # prediction plot task
    plot_predictions_task = PythonOperator(
        task_id="plot_predictions_task",
        python_callable=plot_predictions,
    )
    # send model results to API endpoint
    post_model_results_task = PythonOperator(
        task_id="post_model_results_task",
        python_callable=post_model_results
    )   
            
    # define flow dependencies
    start_task >> api_taskgroup >> [sql_taskgroup, ml_taskgroup]
    sql_taskgroup >> plot_data_task
    ml_taskgroup >> [plot_predictions_task, post_model_results_task]
    

# test the DAG
if __name__ == "__main__":
    dag.test()
