import base64
import os
import re
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from src.settings import (CURRENCY_COUNTRY_MAPPING, DAILY_POSITION_ALL_COLS,
                          NOISE_LEVEL)
from src.utils.calc_utils import split_month
from src.utils.json_encoder import json
from src.utils.threading_utils import processing_threads

RISK_ROOT = os.environ.get('RISK_API_URL') or r'https://risk-api.polymerrisk.com/api/'
PASSWORD = os.environ.get('RISK_API_AUTH_PWD') or r'obiwanhasthehighground'
QADIRECT_ROOT = os.environ.get('QAD_API_URL') or r'https://qadirect-api.polymerquant.com/'
SUPER_USER = os.environ.get('RISK_API_USER') or 'minfengy'


def get_token_payload(token):
    payload_str = token.split('.')[1]
    if not payload_str.endswith('=='):
        payload_str = payload_str + '=='
    token_payload = json.loads(base64.b64decode(payload_str).decode())

    return token_payload


def get_token(
        username,
        password=PASSWORD,
        ip='string',
        loc='string'
) -> str:
    response = requests.post(
        url=RISK_ROOT + 'Auth',
        json={
            'userName': username,
            'password': password,
            'extraUserInfo': {
                'ip': ip,
                'loc': loc,
            }
        }
    )
    if response.ok:
        token = json.loads(response.text).get('token')
        return token


def get_super_token() -> str:
    return get_token(SUPER_USER)


def auth_check(user_token: str) -> bool:
    user_info = get_token_payload(user_token)
    is_admin = False
    if 'admin' in user_info.get('role'):
        is_admin = True

    return is_admin


def reduce_mem_usage(df: DataFrame) -> DataFrame:
    """
    iterate through all the columns of a dataframe and
    modify the data type to reduce memory usage.
    """
    for col in df.columns:
        key_detail = DAILY_POSITION_ALL_COLS.get(col)
        if key_detail is not None:
            col_type = key_detail.get('type')
            if col_type is not None:
                df[col] = df[col].astype(col_type)

    return df


class ReturnType:
    DATA_FRAME = 'DataFrame'
    TEXT = 'text'
    JSON = 'json'


def get_aum(date):
    RISK_ROOT = r'https://risk-api.polymerrisk.com/api/'
    tk_response = requests.post(
        url=RISK_ROOT + 'Auth',
        json={
            'userName': 'minfengy',
            'password': 'obiwanhasthehighground',
        }
    )
    token = json.loads(tk_response.text).get('token')

    url = f'{RISK_ROOT}AumHistories/{date}'

    response = requests.get(
        url=url,
        headers={
            'accept': 'text/plain',
            'Content-Type': 'application/json-patch+json',
            'Authorization': 'Bearer ' + token
        },
    )

    return json.loads(response.text)


class RiskApiDataLoader:
    def __init__(
            self,
            token: str,
            risk_root: str = RISK_ROOT,
            qa_root: str = QADIRECT_ROOT,
    ):
        self._token = token
        if not self._token:
            raise ValueError('Token not valid')

        self._risk_root = risk_root
        self._qa_root = qa_root
        self._header = {
            'accept': 'text/plain',
            'Content-Type': 'application/json-patch+json',
            'Authorization': 'Bearer ' + self._token
        }

    @staticmethod
    def _response_handler(
            response,
            return_type: str = ReturnType.DATA_FRAME
    ):
        """
        :return: null dataframe if type is dataframe otherwise return none
        """
        if return_type == ReturnType.DATA_FRAME:
            if response.ok:
                return reduce_mem_usage(DataFrame(json.loads(response.text)))
            else:
                return DataFrame()
        elif return_type == ReturnType.TEXT and response.ok:
            return json.loads(response.text)

        elif return_type == ReturnType.JSON and response.ok:
            return response.text

    def _get_response(
            self,
            url: str,
            query: dict,
            return_type: str = ReturnType.DATA_FRAME
    ):
        return self._response_handler(
            requests.get(
                url=url,
                headers=self._header,
                params=query
            ), return_type)

    def _post_response(
            self,
            url: str,
            query: Union[dict, list],
            return_type: str = ReturnType.DATA_FRAME
    ):
        return self._response_handler(
            requests.post(
                url=url,
                headers=self._header,
                json=query,
            ), return_type)

    def get_positions_daily(
            self,
            filters: dict = None,
            risk_model_override: str = None,
            **kwargs,
    ) -> DataFrame:
        url = f'{self._risk_root}positions/daily'

        options = deepcopy(filters)

        if options is not None and options.get('columns') is not None:
            options['columns'] = list(
                set(
                    options['columns']
                    + [
                        'instrumentType',
                        'optionType',
                        'isLong',
                        'description',
                        'instrumentSubType',
                        'underBbgFirstName',
                        'underlyingInstrumentId',
                        'countryOfExchange',
                        'countryOfRisk',
                        'localCcy'
                    ]
                )
            )

        result = self._post_response(
            url=url,
            query={
                'filter': options,
                'riskModelOverride': risk_model_override,
            },
            return_type=ReturnType.DATA_FRAME
        )

        if kwargs.get('raw_data', False):
            return result

        if len(result) > 0:
            if 'optionType' in result:
                series_is_long = result['isLong'] == True
                result['isLong'] = np.where(
                    result['optionType'] == 'Put',
                    ~series_is_long,
                    series_is_long
                )

            # if stocks have same name and bbg ticker first name they are the same company
            # they should have same underlyingInstrumentId and underlyingBbgYellowKey
            result['underBbgFirstName'] = result['underlyingBbgYellowKey'].fillna('NA').replace({'': 'NA'}).str.split().str[0]
            for col in ['underlyingBbgYellowKey']:
                result[col] = result.groupby(['description', 'instrumentSubType', 'underBbgFirstName'])[col].transform('first')
                # fill na underlying instrument id and underlying key
                result[col] = result[col].replace({'': np.NaN}).fillna(result['description'])

            result['uniqueId'] = result['description'].astype(str) \
                                 + result['instrumentSubType'].astype(str) \
                                 + result['underBbgFirstName'].astype(str) \
                                 + result['countryOfExchange'].astype(str)
            del result['underBbgFirstName']

            # mapping cash country risk
            result['countryOfRisk'] = np.where(
                result['instrumentSubType'] == 'Cash',
                result['localCcy'].replace(CURRENCY_COUNTRY_MAPPING),
                result['countryOfRisk']
            )
        return result

    def get_positions_daily_by_month(
            self,
            filters: dict = None,
            risk_model_override: str = None,
            **kwargs,
    ) -> DataFrame:
        url = f'{self._risk_root}positions/daily'

        options = deepcopy(filters)

        start_date = filters['startdate']
        end_date = filters['enddate']


        if options is not None and options.get('columns') is not None:
            options['columns'] = list(
                set(
                    options['columns']
                    + [
                        'instrumentType',
                        'optionType',
                        'isLong',
                        'description',
                        'instrumentSubType',
                        'underBbgFirstName',
                        'underlyingInstrumentId',
                        'countryOfExchange',
                        'countryOfRisk',
                        'localCcy'
                    ]
                )
            )

        result = self._post_response(
            url=url,
            query={
                'filter': options,
                'riskModelOverride': risk_model_override,
            },
            return_type=ReturnType.DATA_FRAME
        )

        if kwargs.get('raw_data', False):
            return result

        if len(result) > 0:
            if 'optionType' in result:
                series_is_long = result['isLong'] == True
                result['isLong'] = np.where(
                    result['optionType'] == 'Put',
                    ~series_is_long,
                    series_is_long
                )

            # if stocks have same name and bbg ticker first name they are the same company
            # they should have same underlyingInstrumentId and underlyingBbgYellowKey
            result['underBbgFirstName'] = result['underlyingBbgYellowKey'].fillna('NA').replace({'': 'NA'}).str.split().str[0]
            for col in ['underlyingInstrumentId', 'underlyingBbgYellowKey']:
                result[col] = result.groupby(['description', 'instrumentSubType', 'underBbgFirstName'])[col].transform('first')
                # fill na underlying instrument id and underlying key
                result[col] = result[col].replace({'': np.NaN}).fillna(result['description'])

            result['uniqueId'] = result['description'].astype(str) \
                                 + result['instrumentSubType'].astype(str) \
                                 + result['underBbgFirstName'].astype(str) \
                                 + result['countryOfExchange'].astype(str)
            del result['underBbgFirstName']

            # mapping cash country risk
            result['countryOfRisk'] = np.where(
                result['instrumentSubType'] == 'Cash',
                result['localCcy'].replace(CURRENCY_COUNTRY_MAPPING),
                result['countryOfRisk']
            )
        return result

    def get_trades_filter(
            self,
            instrument_ids: list = None,
            instrument_types: list = None,
            instrument_sub_types: list = None,
            columns: list = None,
            book: str = None,
            pm: str = None,
            sub_pm: str = None,
            from_date: str = None,
            to_date: str = None,
            date: str = None,
            port: str = None,
            remove_internal: bool = True,
    ) -> DataFrame:
        url = f'{self._risk_root}trades/filter'

        if remove_internal and columns is not None:
            columns = list(set(columns + ['counterparty', 'counterpartyShortName', 'instrumentType', 'strategy']))

        trades = self._post_response(
            url=url,
            query={
                'instrumentIds': instrument_ids,
                'instrumentTypes': instrument_types,
                'instrumentSubTypes': instrument_sub_types,
                'columns': columns,
                'book': book,
                'pm': pm,
                'subPm': sub_pm,
                'from': from_date,
                'to': to_date,
                'date': date,
                'port': port
            }
        )

        if len(trades) > 0:
            trades['tradeDate'] = trades['tradeDate'].str[:10]

            if remove_internal:
                instrumentFilter = [
                    'Repurchase Agreement', 'Cash', 'FX Forward', 'FX Spot', 'FX Swap']

                trades = trades[~trades.strategy.str.contains("PTH")]
                trades = trades[~trades.instrumentType.isin(instrumentFilter)]

                for key in ['INTERNAL', 'REORG']:
                    trades = trades[~trades['counterparty'].str.contains(key, flags=re.IGNORECASE)]
                for key in ['INT', 'CROSS']:
                    trades = trades[~trades['counterpartyShortName'].str.contains(key, flags=re.IGNORECASE)]

        return trades

    def get_pnls(
            self,
            start_date: str,
            end_date: str,
            book: str = None,
            pm: str = None,
            sub_pm: str = None,
            group_by: str = None,
            book_group: str = None,
            include_bp: bool = True,
            include_aum: bool = True,
            port: str = None,
    ) -> DataFrame:
        url = f'{self._risk_root}pnls'

        result = self._post_response(
            url=url,
            query={
                'dates': [start_date, end_date],
                'book': book,
                'pm': pm,
                'subPm': sub_pm,
                'groupBy': group_by,
                'bookGroup': book_group,
                'includeBp': include_bp,
                'includeAum': include_aum,
                'port': port
            },
            return_type=ReturnType.DATA_FRAME
        )

        if len(result) > 0:
            if include_aum and 'aum' not in result:
                result['aum'] = 0
                result['preAum'] = 0
                result['returnAum'] = 0
            if 'bp' not in result:
                result['bp'] = 0
                result['preBp'] = 0
                result['returnBp'] = 0

        return result

    def get_barra_pnls(
            self,
            from_date: str = None,
            to_date: str = None,
            port: str = None,
            analysis_settings: str = None,
            include_scaled_rtn: bool = True,
    ):
        url = f'{self._risk_root}BpmResults/PortfolioFactorReturn'

        result = self._post_response(
            url=url,
            query={
                'from': from_date,
                'to': to_date,
                'port': port,
                'analysisSettings': analysis_settings,
                'includeScaledRtn': include_scaled_rtn,
            },
            return_type=ReturnType.DATA_FRAME,
        )

        result['pm'] = port

        if 'date' in result:
            result['date'] = pd.to_datetime(
                result['date'],
                format='%Y-%m-%d'
            ).apply(lambda x: x.strftime('%Y-%m-%d'))

        result.replace('NaN', np.NaN, inplace=True)

        return result

    def get_bpm_result(
            self,
            report_type: str,
            port: str = None,
            from_date: str = None,
            to_date: str = None,
            date: str = None,
            analysis_settings: str = None,
            pm: str = None,
    ):
        return self._get_response(
            url=f'{self._risk_root}bpmResults',
            query={
                'port': port,
                'from': from_date,
                'to': to_date,
                'date': date,
                'analysisSettings': analysis_settings,
                'reportType': report_type,
                'pm': pm,
            },
            return_type=ReturnType.TEXT
        )

    def get_bpm_factor_return(
            self,
            port: str = None,
            from_date: str = None,
            to_date: str = None,
            analysis_settings: str = None,
            include_scaled_rtn: bool = True,
    ):
        return self._post_response(
            url=f'{self._risk_root}BpmResults/RawFactorReturn',
            query={
                'port': port,
                'from': from_date,
                'to': to_date,
                'analysisSettings': analysis_settings,
                'includeScaledRtn': include_scaled_rtn,
            },
            return_type=ReturnType.DATA_FRAME
        )

    def get_model_direct(
            self,
            report_type: str,
            from_date: str = None,
            to_date: str = None,
            ref_date: str = None,
            model: str = None,
            output: str = ReturnType.JSON
    ):
        return self._get_response(
            url=f'{self._risk_root}ModelDirectData',
            query={
                'from': from_date,
                'to': to_date,
                'refDate': ref_date,
                'reportType': report_type,
                'model': model,
            },
            return_type=output
        )

    def get_books(self):
        return self._get_response(
            url=f'{self._risk_root}books',
            query={},
            return_type=ReturnType.DATA_FRAME
        )

    def get_daily_summary(
            self,
            book: str = None,
            from_date: str = None,
            to_date: str = None,
            instrument_types: list = None,
            columns: list = None,
            remove_inactive=False,
    ) -> DataFrame:
        result = self._post_response(
            url=f'{self._risk_root}dailysummary',
            query={
                'book': book,
                'from': from_date,
                'to': to_date,
                'instrumentTypes': instrument_types,
                'columns': columns
            },
            return_type=ReturnType.DATA_FRAME
        )

        if 'positionDate' in result:
            result['positionDate'] = result['positionDate'].str[:10]

        if remove_inactive and 'deltaUsd' in result and 'dailyPnlUsd' in result:
            result = result[
                (result['deltaUsd'].abs() > NOISE_LEVEL)
                & (result['dailyPnlUsd'].abs() > NOISE_LEVEL)
                ]

        return result

    def get_daily_summary_monthly(
            self,
            book: str = None,
            from_date: str = None,
            to_date: str = None,
            instrument_types: list = None,
            columns: list = None,
            remove_inactive=False,
            create_unique_id=True
    ) -> DataFrame:
        if create_unique_id and columns is not None:
            columns = list(set(columns + ['countryOfExchange', 'instrumentSubType', 'description']))

        monthly_dates = split_month(
            start_date=from_date,
            end_date=to_date
        )

        def query_func(pos, options):
            response = self.get_daily_summary(**options)
            if len(response) > 0:
                pos.append(response)

        positions = []
        options_by_date = []
        for item in monthly_dates:
            options_by_date.append([
                positions,
                {
                    'book': book,
                    'from_date': item[0],
                    'to_date': item[1],
                    'instrument_types': instrument_types,
                    'columns': columns,
                    'remove_inactive': remove_inactive,
                }])

        processing_threads(
            target=query_func,
            args=options_by_date,
        )

        if len(positions) == 0:
            return DataFrame()

        positions = pd.concat(positions)

        if create_unique_id:
            # if stocks have same name and bbg ticker first name they are the same company
            positions['underBbgFirstName'] = positions['underlyingBbgYellowKey'].fillna('NA').replace({'': 'NA'}).str.split().str[0]
            positions['underlyingInstrumentId'] = positions.groupby(['description', 'instrumentSubType', 'underBbgFirstName'], observed=True)['underlyingInstrumentId'].transform(
                'first')
            positions['underlyingBbgYellowKey'] = positions.groupby(['description', 'instrumentSubType', 'underBbgFirstName'], observed=True)['underlyingBbgYellowKey'].transform(
                'first')
            positions['uniqueId'] = positions['description'].astype(str) + positions['instrumentSubType'].astype(str) + positions['underBbgFirstName'].astype(str) + positions[
                'countryOfExchange'].astype(str)
            del positions['underBbgFirstName']
            # fill na underlying instrument id and underlying key
            positions['underlyingInstrumentId'] = positions['underlyingInstrumentId'].replace({'': np.NaN}).fillna(positions['description'])
            positions['underlyingBbgYellowKey'] = positions['underlyingBbgYellowKey'].replace({'': np.NaN}).fillna(positions['description'])
        return positions

    def get_barra_id_from_bbg(
            self,
            query_list: list = None
    ):
        url = f'{self._risk_root}BarraIds/toBarraIds'

        result = self._post_response(
            url=url,
            query=query_list,
            return_type=ReturnType.TEXT
        )
        if result is not None and len(result) > 0:
            output_list = []
            for i in result:
                if i['hasError'] is False:
                    output_list.append({
                        'barraId': i['barraId'],
                        'bbgYellowKey': i['query']['inputId']
                    })
            result = DataFrame(output_list)
            return result

    def get_down_corr(
            self,
            start_date: str,
            end_date: str,
            books: list = None,
            min_data_points: int = 20,
            benchmark_model: str = None,
            index: str = None,
            group_type: int = 0
    ):
        return self._post_response(
            url=f'{self._risk_root}Pnls/correlations/DownMarket',
            query={
                'startDate': start_date,
                'endDate': end_date,
                'index': index,
                'books': books,
                'minDataPoints': min_data_points,
                'benchmarkModel': benchmark_model,
                'groupType': group_type,
            },
            return_type=ReturnType.DATA_FRAME,

        )

    def get_pnls_calc(
            self,
            from_date: str,
            to_date: str,
            group_types: list = None,
            books: list = None
    ):
        return self._post_response(
            url=f'{self._risk_root}Pnls/calc',
            query={
                'from': from_date,
                'to': to_date,
                'groupTypes': group_types,
                'books': books,
            },
            return_type=ReturnType.TEXT,
        )

    def get_supported_index(self):
        return self._get_response(
            url=f'{self._risk_root}Pnls/correlations/SupportedIndexOptions',
            query={},
            return_type=ReturnType.DATA_FRAME
        )

    def get_qa_index_price(
            self,
            rics: Union[str, list],
            start_date: str,
            end_date: str,
            result_dates_descending_order: bool = False
    ):
        url = f'{self._qa_root}quote/historicalIndexQuotes'

        if isinstance(rics, str):
            ric_input = [rics]
        else:
            ric_input = rics

        result = self._post_response(
            url=url,
            query={
                'rics': ric_input,
                'startDate': start_date,
                'endDate': end_date,
                'resultDatesDescendingOrder': result_dates_descending_order,
            },
            return_type=ReturnType.TEXT
        )

        if result.get('quotes') is not None:
            result = result['quotes']
            if len(result) > 0:
                for item in result:
                    quotes = DataFrame(item['quotes'])
                    quotes.replace({None: np.NaN}, inplace=True)
                    if (~quotes['returnIndex'].isna()).sum() == 0:
                        quotes['returnIndex'] = quotes['priceIndex']
                    quotes['valueDate'] = quotes['valueDate'].str[:10]

                    item['quotes'] = quotes

                return result

    def get_qa_stock_quotes(
            self,
            rics: Union[str, list],
            start_date: str,
            end_date: str,
            adjustment_ref_date: str = None,
            primary_quotes_only: bool = True,
            apply_backdated_adjustment: bool = True,
            result_dates_descending_order: bool = False
    ) -> DataFrame:
        url = f'{self._qa_root}quote/historicalQuotes'

        if isinstance(rics, str):
            ric_input = [rics]
        else:
            ric_input = rics

        batch_size = 100
        batched_ric_list = [ric_input[i:i + batch_size] for i in range(0, len(ric_input), batch_size)]

        qad_price_results = processing_threads(
            self._post_response,
            [(url,
              {
                'rics': batched_rics,
                'startDate': start_date,
                'endDate': end_date,
                'adjustmentRefDate': adjustment_ref_date,
                'primaryQuotesOnly': primary_quotes_only,
                'applyBackdatedAdjustment': apply_backdated_adjustment,
                'resultDatesDescendingOrder': result_dates_descending_order,
              }) for batched_rics in batched_ric_list]
        )

        api_quotes = []
        for res in qad_price_results:
            if res is not None and res.get('quotes') is not None:
                api_quotes.extend(res['quotes'])

        if len(api_quotes) > 0:
            df = []
            for item in api_quotes:
                df_ticker = pd.DataFrame(item['quotes'])
                df_ticker['ric'] = item['ric']
                df.append(df_ticker)
            df = pd.concat(df)
            df['marketDate'] = df['marketDate'].str[:10]
            return df

    def get_qa_stock_quotes_start_end_date(
            self,
            rics: Union[str, list],
            start_date: str,
            end_date: str,
            adjustment_ref_date: str,
            primary_quotes_only: bool = True,
            apply_backdated_adjustment: bool = True,
            result_dates_descending_order: bool = False
    ):
        url = f'{self._qa_root}quote/historicalQuotesForStartDateAndEndDate'

        if isinstance(rics, str):
            ric_input = [rics]
        else:
            ric_input = rics

        result = self._post_response(
            url=url,
            query={
                'rics': ric_input,
                'startDate': start_date,
                'endDate': end_date,
                'adjustmentRefDate': adjustment_ref_date,
                'primaryQuotesOnly': primary_quotes_only,
                'applyBackdatedAdjustment': apply_backdated_adjustment,
                'resultDatesDescendingOrder': result_dates_descending_order,
            },
            return_type=ReturnType.TEXT
        )

        if result is not None and result.get('quotes') is not None:
            result = result['quotes']
            if len(result) > 0:
                return result
