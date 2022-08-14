import os
from typing import List

import boto3
import pandas as pd
from pydantic import BaseModel, constr
from datetime import datetime, timedelta
import time


AWS_REGION_END_POINT = 'ap-northeast-1'
AWS_STOCHASTIC_SIMULATION_JOB_Q = 'stochastic-simulation-job-q'
AWS_STOCHASTIC_SIMULATION_JOB_DEF = 'stochastic-simulation-def'

AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')


class JobRes(BaseModel):
    jobArn: str
    jobName: str
    jobId: str
    createdAt: int = None
    status: str = None


class JobLogModel(BaseModel):
    timestamp: str = None
    message: str = None


def get_aws_batch_client():
    return boto3.client(
        'batch',
        region_name=AWS_REGION_END_POINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )


def get_clock_watch_client():
    return boto3.client(
        'logs',
        region_name=AWS_REGION_END_POINT,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )


def submit_aws_batch_jobs(
        job_name: constr(max_length=128),
        job_queue: str,
        job_definition: str,
        cmd_path: str,
        param_list: List[str],
) -> JobRes:
    batch_client = get_aws_batch_client()

    req = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': job_definition,
        'containerOverrides': {
            'command': ['python', cmd_path] + param_list
        },
    }

    res = batch_client.submit_job(**req)
    job_res = JobRes(
        jobArn=res['jobArn'],
        jobName=res['jobName'],
        jobId=res['jobId'],
    )

    return job_res


def get_aws_batch_log_stream_name(job_id: str):
    batch_client = get_aws_batch_client()
    des = batch_client.describe_jobs(jobs=[job_id])

    if des and des.get('jobs'):
        job = des['jobs'][0]
        if job.get('container') and job['container'].get('logStreamName'):
            return job['container']['logStreamName']


def query_clock_watch_logs(log_stream_name: str):
    client = get_clock_watch_client()

    query = f'fields @timestamp, @message, @logStream ' \
            f'| filter @logStream == "{log_stream_name}" ' \
            f'| sort @timestamp desc ' \
            f'| limit 20'

    log_group = '/aws/batch/return-based-risk-management'

    start_query_response = client.start_query(
        logGroupName=log_group,
        startTime=int((datetime.today() - timedelta(days=365)).timestamp()),
        endTime=int(datetime.now().timestamp()),
        queryString=query,
    )

    query_id = start_query_response['queryId']

    response = None

    while response == None or response['status'] == 'Running':
        time.sleep(1)
        response = client.get_query_results(
            queryId=query_id
        )

    res = []
    if response and len(response['results']) > 0:
        for item in response['results']:
            msg = pd.DataFrame(item)
            msg.index = msg['field']
            del msg['field']
            res.append(msg.T)
        res = pd.concat(res)
        del res['@ptr']
        res['timestamp'] = res['@timestamp']
        res['message'] = res['@message']
        return res[['timestamp', 'message']]
