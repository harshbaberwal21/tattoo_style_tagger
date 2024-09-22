from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from lib.drivers.data_driver import(
read_tattoos_meta_data_driver,
process_tattoos_meta_data_driver,
download_tattoos_images_driver,
pre_process_tattoos_images_driver,
augment_tattoos_images_data_driver,
)

# Define or Instantiate DAG
dag = DAG(
    'tattoo_style_tagger_data_prep'
)


read_tattoos_meta_data_task = PythonOperator(
    task_id='read_tattoos_meta_data',
    python_callable=read_tattoos_meta_data_driver,
    dag=dag,
)


process_tattoos_meta_data_task = PythonOperator(
    task_id='process_tattoos_meta_data',
    python_callable=process_tattoos_meta_data_driver,
    dag=dag,
)


download_tattoos_images_task = PythonOperator(
    task_id='download_tattoos_images',
    python_callable=download_tattoos_images_driver,
    dag=dag,
)


compress_and_backup_images_task = BashOperator(
    task_id='compress_and_backup_images',
    bash_command='',
    cwd='/tmp',
    dag=dag,
)


pre_process_tattoos_images_task = PythonOperator(
    task_id='pre_process_tattoos_images',
    python_callable=pre_process_tattoos_images_driver,
    dag=dag,
)


augment_tattoos_images_data_task = PythonOperator(
    task_id='augment_tattoos_images_data',
    python_callable=augment_tattoos_images_data_driver,
    dag=dag,
)


(
    read_tattoos_meta_data_task
    >> process_tattoos_meta_data_task
    >> download_tattoos_images_task
    >> compress_and_backup_images_task
    >> pre_process_tattoos_images_task
    >> augment_tattoos_images_data_task
)