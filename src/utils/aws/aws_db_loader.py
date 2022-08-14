import datetime
import json
import os
from decimal import Decimal
from typing import List

import boto3
import jax.numpy as jnp
import numpy as np
import pandas as pd
from boto3.dynamodb.conditions import Key

from src.models.return_based_analysis.simulator_model import (
    SimPageSettings,
    SimJobResponse,
    SimulationJobType,
)
from src.utils.logger import logger

AWS_REGION_END_POINT = 'ap-east-1'
AWS_REGION_END_POINT_TKY = 'ap-northeast-1'

COMMISSION_TABLE = 'commission_pb_mapping'

COMMISSION_TARGET_TABLE = 'commission_target'
HFRX_DATA_TABLE = 'hfrx_index_return'
HFRX_INDEX_NAME_TABLE = 'hfrx_index_name'
SIMULATOR_TABLE = 'stochastic_simulation_result'
STOCK_MKT_RT_TABLE = 'stock_market_return'
LIQUIDITY_IGNORE_TABLE = 'liquidity_ticker_ignore'
POS_ATTR_OVERRIDE_TABLE = 'position_attr_override'
POS_MASTER_USER_STATE_TABLE = 'pos_master_user_state'

S3_RETURN_BASED_RISK_MGMT = 'return-based-risk-mgmt'

AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.environ.get('AWS_SECRET_KEY')


def encoder(obj):
    if isinstance(obj, jnp.float32):
        return float(obj)
    if obj in [np.nan, jnp.nan]:
        return None


def get_aws_db(region=AWS_REGION_END_POINT):
    return boto3.resource(
        'dynamodb',
        region_name=region,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )


def get_aws_s3(region=AWS_REGION_END_POINT_TKY):
    return boto3.client(
        's3',
        region_name=region,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
    )


def read_pb_mapping():
    dynamodb = get_aws_db()
    table = dynamodb.Table(COMMISSION_TABLE)

    response = table.scan()

    res = pd.DataFrame(response['Items'])

    # only keep latest update
    if len(res):
        res.drop_duplicates(subset=['shortName'], inplace=True)

    return res


def read_target_alloc(year: str) -> dict:
    dynamodb = get_aws_db()
    table = dynamodb.Table(COMMISSION_TARGET_TABLE)

    def get_res(yr):
        res_list = []
        done = False
        start_key = None
        while not done:
            if start_key:
                response = table.query(
                    KeyConditionExpression=Key('year').eq(str(yr)),
                    ExclusiveStartKey=start_key,
                )
            else:
                response = table.query(
                    KeyConditionExpression=Key('year').eq(str(yr)),
                )
            start_key = response.get('LastEvaluatedKey', None)
            done = start_key is None
            res_list = res_list + response['Items']
        return res_list

    res = get_res(year)

    if len(res) == 0:
        pre_year = str(int(year) - 1)
        res = get_res(pre_year)

    if len(res) > 0:
        res = sorted(res, key=lambda x: x['update_time'])
        res_item = res[-1]
        res_item['target'] = json.loads(res_item['target'], parse_float=Decimal)
        return res_item
    else:
        return {
            'year': year,
            'update_time': None,
            'target': {},
        }


def upload_target_alloc(
        target_json: str,
        year: str,
        user: str,
        update_time: str
) -> str:
    dynamodb = get_aws_db()

    # upload data to db
    table = dynamodb.Table(COMMISSION_TARGET_TABLE)

    try:
        with table.batch_writer() as batch:
            item_list = [{
                'year': year,
                'update_time': update_time,
                'target': target_json,
                'user': user,
            }]
            for item in item_list:
                batch.put_item(Item=item)
        return 'Updating succeeded'
    except Exception as e:
        logger.exception(e)
        return 'Updating failed'


def upload_hfrx_data(start_date, end_date):
    dynamodb = get_aws_db()
    # read returns from excel
    items = pd.read_csv(
        r'C:\Users\minfengy\Documents\hfrx_daily_index_data.csv',
        skiprows=4,
        skipfooter=5,
        names=['date', 'index', 'shortName', 'return', 'indexValue'],
        header=None,
    )

    items['return'] = items['return'].str.rstrip('%').astype(float) / 100

    # upload data to db
    table_data = dynamodb.Table(HFRX_DATA_TABLE)
    items['date'] = pd.to_datetime(items['date']).apply(lambda x: x.strftime('%Y-%m-%d'))
    items = items[items['date'].between(start_date, end_date)]

    with table_data.batch_writer() as batch:
        item_list = json.loads(json.dumps(items.to_dict(orient='records')), parse_float=Decimal)
        for item in item_list:
            batch.put_item(Item=item)

    # upload index name to db
    table_index_name = dynamodb.Table(HFRX_INDEX_NAME_TABLE)
    with table_index_name.batch_writer() as batch:
        for nm in list(set(list(items['index']))):
            batch.put_item(Item={
                'index': nm,
            })


def read_hfrx_data(start_date, end_date, index):
    dynamodb = get_aws_db()
    table = dynamodb.Table(HFRX_DATA_TABLE)

    response = table.query(
        KeyConditionExpression=Key('index').eq(index) & Key('date').between(start_date, end_date)
    )

    res = pd.DataFrame(response['Items'])
    res.sort_values(by='date', inplace=True)
    return res


def read_hfrx_index_name():
    dynamodb = get_aws_db()
    table = dynamodb.Table(HFRX_INDEX_NAME_TABLE)

    response = table.scan()

    res = pd.DataFrame(response['Items'])
    return res


class AwsSimulationClient:
    def __init__(self):
        self._s3_client = get_aws_s3()
        dynamodb = get_aws_db(region=AWS_REGION_END_POINT_TKY)
        self._dynamodb_table = dynamodb.Table(SIMULATOR_TABLE)

    def upload_simulation_result(
            self,
            db_input: SimJobResponse
    ) -> any:
        """ save result json to S3 """
        if not db_input.jobDisplayName:
            db_input.jobDisplayName = db_input.jobName

        # upload data to s3
        response = self._s3_client.put_object(
            Body=db_input.json(encoder=encoder),
            Bucket=S3_RETURN_BASED_RISK_MGMT,
            Key=f'{db_input.jobType}/{db_input.jobName}',
        )

        return response

    def create_simulation_settings(
            self,
            db_input: SimJobResponse
    ) -> str:
        try:
            if not db_input.jobDisplayName:
                db_input.jobDisplayName = db_input.jobName

            response = self._dynamodb_table.put_item(
                Item=json.loads(db_input.json(encoder=encoder), parse_float=Decimal)
            )
            return 'Job created'
        except Exception as e:
            logger.exception(e)
            return 'Job creation failed'

    def upload_simulation_settings(
            self,
            db_input: SimJobResponse
    ) -> str:
        """
        upload job name/type/setting to dynamo db for quick query
        :param db_input:
        :return:
        """
        try:
            key = {
                'jobName': db_input.jobName,
                'jobType': db_input.jobType,
            }

            db_input_dict = json.loads(db_input.json(), parse_float=Decimal)

            expression_attribute_values = {}
            update_expression_list = []
            non_primary_keys = [i for i in list(db_input_dict.keys()) if i not in ['jobName', 'jobType'] and i is not None]
            for col in non_primary_keys:
                col_val = db_input_dict.get(col)
                if type(col_val) == float:
                    col_val = Decimal(col_val)
                if col_val:
                    expression_attribute_values[f':{col}'] = col_val
                    update_expression_list.append(f'{col}=:{col}')

            self._dynamodb_table.update_item(
                Key=key,
                UpdateExpression='set ' + ', '.join(update_expression_list),
                ExpressionAttributeValues=expression_attribute_values,
                ReturnValues="UPDATED_NEW"
            )
            return 'Updating succeeded'
        except Exception as e:
            logger.exception(e)
            return 'Updating failed'

    def load_simulation_result(
            self,
            job_name: str,
            job_type: SimulationJobType,
    ) -> SimJobResponse:
        obj = self._s3_client.get_object(
            Bucket=S3_RETURN_BASED_RISK_MGMT,
            Key=f'{job_type}/{job_name}'
        )

        res = SimJobResponse.parse_obj(json.loads(obj['Body'].read().decode('utf-8')))
        return res

    def query_simulation_result_by_factors(
            self,
            job_type: SimulationJobType,
            params_page_settings: SimPageSettings,
            port: str = None
    ) -> List[SimJobResponse]:
        # read a list of simulation result by job type and factor settings

        res = []
        done = False
        start_key = None
        while not done:

            query_kwd = {
                'KeyConditionExpression': Key('jobType').eq(str(job_type.value)),
                'IndexName': 'jobTypeIndex',
                # 'ProjectionExpression': ', '.join(col_list),
            }

            if start_key:
                query_kwd['ExclusiveStartKey'] = start_key

            response = self._dynamodb_table.query(**query_kwd)
            start_key = response.get('LastEvaluatedKey', None)
            done = start_key is None
            res = res + response['Items']

        output = []
        for item in res:
            obj = SimJobResponse.parse_obj(item)
            if obj.simPageSettings:
                # return only results match factor settings
                factor_req = sorted([i.name for i in params_page_settings.factorSettings.factors])
                factor_db = sorted([i.name for i in obj.simPageSettings.factorSettings.factors])

                if factor_req == factor_db:
                    if not port or (port and port == obj.simPageSettings.portSettings.port):
                        output.append(obj)

        return output

    def query_simulation_result_by_job_name(
            self,
            job_name: str,
    ) -> SimJobResponse:
        # read a list of simulation result by job type and factor settings

        res = []
        done = False
        start_key = None
        while not done:
            query_kwd = {
                'KeyConditionExpression': Key('jobName').eq(job_name),
            }

            if start_key:
                query_kwd['ExclusiveStartKey'] = start_key

            response = self._dynamodb_table.query(**query_kwd)
            start_key = response.get('LastEvaluatedKey', None)
            done = start_key is None
            res = res + response['Items']

        if len(res) > 0:
            obj = SimJobResponse.parse_obj(res[0])
            return obj

    def delete_simulation_result(
            self,
            job_name: str,
            job_type: SimulationJobType,
    ):
        self._delete_simulation_result_s3(
            job_name=job_name,
            job_type=job_type
        )
        self._delete_simulation_result_dynamo(
            job_name=job_name,
            job_type=job_type
        )

    def _delete_simulation_result_s3(
            self,
            job_name: str,
            job_type: SimulationJobType,
    ) -> str:
        try:
            response = self._s3_client.delete_object(
                Bucket=S3_RETURN_BASED_RISK_MGMT,
                Key=f'{job_type}/{job_name}'
            )

            return 'Delete succeeded'
        except Exception as e:
            logger.exception(e)
            return 'Delete failed'

    def _delete_simulation_result_dynamo(
            self,
            job_name: str,
            job_type: SimulationJobType
    ) -> str:
        try:
            key = {
                'jobName': job_name,
                'jobType': job_type,
            }

            self._dynamodb_table.delete_item(
                Key=key,
            )
            return 'Delete succeeded'
        except Exception as e:
            logger.exception(e)
            return 'Delete failed'


def read_market_return(date):
    dynamodb = get_aws_db(region=AWS_REGION_END_POINT_TKY)
    table = dynamodb.Table(STOCK_MKT_RT_TABLE)

    res = []
    done = False
    start_key = None
    while not done:
        query_kwd = {
            'KeyConditionExpression': Key('date').eq(str(date)),
            'IndexName': 'date-index',
        }

        if start_key:
            query_kwd['ExclusiveStartKey'] = start_key

        response = table.query(**query_kwd)
        start_key = response.get('LastEvaluatedKey', None)
        done = start_key is None
        res = res + response['Items']

    res = pd.DataFrame(res)

    return res


def read_market_return_abnormal(date, threshold=0.1):
    dynamodb = get_aws_db(region=AWS_REGION_END_POINT_TKY)
    table = dynamodb.Table(STOCK_MKT_RT_TABLE)

    res = []
    abnormal_condition = [Key('dailyRt').gte(Decimal(str(threshold))), Key('dailyRt').lte(Decimal(str(-threshold)))]

    for cd in abnormal_condition:
        done = False
        start_key = None
        while not done:
            query_kwd = {
                'KeyConditionExpression': Key('month').eq(date[:7]) & cd,
                # 'FilterExpression': cd,
                'IndexName': 'month-dailyRt-index',
            }

            if start_key:
                query_kwd['ExclusiveStartKey'] = start_key
            response = table.query(**query_kwd)
            start_key = response.get('LastEvaluatedKey', None)
            done = start_key is None
            res = res + response['Items']
    res = pd.DataFrame(res)

    return res


def upload_liquidity_ignore_tk(
        user,
        ticker,
        add,
) -> str:
    dynamodb = get_aws_db(region=AWS_REGION_END_POINT_TKY)

    # upload data to db
    table_data = dynamodb.Table(LIQUIDITY_IGNORE_TABLE)

    key = {
        'ticker': ticker,
    }

    try:
        if add:
            table_data.update_item(
                Key=key,
                UpdateExpression='set userName=:userName, updateTime=:updateTime',
                ExpressionAttributeValues={
                    ':userName': user,
                    ':updateTime': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                },
                ReturnValues="UPDATED_NEW"
            )
            return 'Ignore Ticker Updated'
        else:
            table_data.delete_item(
                Key=key,
            )
            return 'Ignore Ticker Deleted'
    except Exception as e:
        logger.exception(e)
        return 'Ignore Ticker Update Failed'


def read_liquidity_ignore_tk() -> pd.DataFrame:
    dynamodb = get_aws_db(region=AWS_REGION_END_POINT_TKY)
    table = dynamodb.Table(LIQUIDITY_IGNORE_TABLE)

    response = table.scan()

    res = pd.DataFrame(response['Items'])
    return res


def upload_pos_attr_override(
        user,
        ticker,
        attr,
        value_override,
        add,
) -> str:
    dynamodb = get_aws_db(region=AWS_REGION_END_POINT_TKY)

    # upload data to db
    table_data = dynamodb.Table(POS_ATTR_OVERRIDE_TABLE)

    key = {
        'ticker': ticker,
    }

    try:
        if add:
            table_data.update_item(
                Key=key,
                UpdateExpression='set userName=:userName, updateTime=:updateTime, attr=:attr, valueOverride=:valueOverride',
                ExpressionAttributeValues={
                    ':userName': user,
                    ':attr': attr,
                    ':valueOverride': value_override,
                    ':updateTime': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                },
                ReturnValues="UPDATED_NEW"
            )
            return 'Pos attr updated'
        else:
            table_data.delete_item(
                Key=key,
            )
            return 'Pos attr deleted'
    except Exception as e:
        logger.exception(e)
        return 'Pos attr Update Failed'


def read_pos_attr_override() -> pd.DataFrame:
    dynamodb = get_aws_db(region=AWS_REGION_END_POINT_TKY)
    table = dynamodb.Table(POS_ATTR_OVERRIDE_TABLE)

    response = table.scan()

    res = pd.DataFrame(response['Items'])
    return res


def upload_user_state(
        user,
        name,
        state_json,
        add,
) -> str:
    dynamodb = get_aws_db(region=AWS_REGION_END_POINT_TKY)

    # upload data to db
    table_data = dynamodb.Table(POS_MASTER_USER_STATE_TABLE)

    key = {
        'userName': user,
        'stateName': name,
    }

    try:
        if add:
            table_data.update_item(
                Key=key,
                UpdateExpression='set updateTime=:updateTime, stateJson=:stateJson',
                ExpressionAttributeValues={
                    ':stateJson': state_json,
                    ':updateTime': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                },
                ReturnValues="UPDATED_NEW"
            )
            return 'User state updated'
        else:
            table_data.delete_item(
                Key=key,
            )
            return 'User state deleted'
    except Exception as e:
        logger.exception(e)
        return 'User state Update Failed'


def read_aws_user_state(user) -> pd.DataFrame:
    dynamodb = get_aws_db(region=AWS_REGION_END_POINT_TKY)
    table = dynamodb.Table(POS_MASTER_USER_STATE_TABLE)

    res = []
    done = False
    start_key = None
    while not done:
        query_kwd = {
            'KeyConditionExpression': Key('userName').eq(str(user)),
        }

        if start_key:
            query_kwd['ExclusiveStartKey'] = start_key

        response = table.query(**query_kwd)
        start_key = response.get('LastEvaluatedKey', None)
        done = start_key is None
        res = res + response['Items']

    return res
