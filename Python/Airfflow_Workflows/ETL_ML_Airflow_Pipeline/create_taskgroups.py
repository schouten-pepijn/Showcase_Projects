from airflow.utils.task_group import TaskGroup
from airflow.operators.python import PythonOperator

from general_functions import (extract_data,
                               convert_data,
                               validate_data,
                               create_sql_table,
                               store_to_sql_table)

from ml_functions import (create_features,
                          train_eval_model)

# create api related tasks
def create_api_taskgroup(group_id):
    with TaskGroup(group_id=group_id) as tq:
        # extract from kraken api
        extract_data_task = PythonOperator(
            task_id='extract_data_task',
            python_callable=extract_data
        )
        # convert to json
        convert_data_task = PythonOperator(
            task_id='convert_data_task',
            python_callable=convert_data
        )
        # validate data types with pydantic
        validate_data_task = PythonOperator(
            task_id='validate_data_task',
            python_callable=validate_data
        )
        # define flow dependencies
        extract_data_task >> convert_data_task >> validate_data_task
    return tq

# create sql related tasks
def create_sql_taskgroup(group_id):
    with TaskGroup(group_id=group_id) as tq:
        # create sql database task
        create_sql_table_task = PythonOperator(
            task_id='create_sql_table_task',
            python_callable=create_sql_table
        )
        # update sql database task
        store_to_sql_task = PythonOperator(
            task_id='store_to_sql_table_task',
            python_callable=store_to_sql_table
        )
        # define flow dependencies
        create_sql_table_task >> store_to_sql_task
    return tq

# create machine learning related tasks
def create_ml_taskgroup(group_id):
    with TaskGroup(group_id=group_id) as tq:
        # create features task
        create_features_task = PythonOperator(
            task_id="create_features_task",
            python_callable=create_features
        )
        # train and evaluate ml model task
        train_eval_model_task = PythonOperator(
            task_id="train_eval_model_task",
            python_callable=train_eval_model
        )
        # define flow dependencies
        create_features_task >> train_eval_model_task
    return tq

