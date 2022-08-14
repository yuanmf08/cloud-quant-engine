from time import time
import pandas as pd
import numpy as np
import requests
import networkx as nx
from datetime import datetime, timedelta

from src.settings import INSTRUMENT_TYPE_WHITE_LIST
from src.utils.threading_utils import processing_threads
from src.utils.db_loader import RiskApiDataLoader


class CorrelationCalculator:
    SEC_MASTER_ENDPOINT: str = 'https://3pwpwxu663.execute-api.ap-northeast-1.amazonaws.com/prod/security-master/mapping'

    def __init__(self, token, book: str = '', output_file: bool = False) -> None:
        self.token = token
        self.book = book
        self.output_file = output_file

    def book_correlations(self, val_date: str, is_long: bool = True, ref_price_days: int = 90, threshold: float = 0.85):
        result = pd.DataFrame()

        print(f' ---- book : {self.book} ---- ')
        try:
            port = self._get_polymer_pos(val_date).reset_index().rename(columns={'UTicker': 'UnderlyingRic'})
        except:
            print(f'book {self.book} not found ... ')
            return

        tmp = self._high_corr_grouping(port, is_long, val_date, ref_price_days, threshold)
        result = result.append(tmp)

        if self.output_file:
            result.to_csv(r'pm_correlated_names_{}_2.csv'.format(val_date))

        return result

    def _security_master_query(self, external_ids: list, external_id_type: str, target_columns: list):
        data = {
            'external_id_type': external_id_type,
            'columns': target_columns,
            'external_ids': external_ids,
        }
        resp = requests.post(self.SEC_MASTER_ENDPOINT, json=data)
        if not resp.ok:
            raise ValueError(f'code {resp.status_code}: {resp.json()}')

        return pd.DataFrame(resp.json())

    def _bbkey_to_ric(self, list_bbkey: list):
        df = self._security_master_query(external_ids=list_bbkey, external_id_type='bbg_yellow_key_composite', target_columns=['ric'])
        df = df.rename(columns = {'bbg_yellow_key_composite':'UnderlyingBbgYellowKey'})
        return  df

    def _qad_price(self, rics, start_date, end_date, msg = False):
        df = pd.DataFrame()

        try:
            api_response = RiskApiDataLoader(token=self.token).get_qa_stock_quotes(
                rics=rics,
                start_date=start_date,
                end_date=end_date,
            )

            if api_response:
                for obj in api_response:
                    quotes = obj.get('quotes', [])

                    if len(quotes) > 0:
                        px_series = pd.Series(
                            [quote.get('closeAdjusted') for quote in quotes],
                            index=[quote.get('marketDate') for quote in quotes],
                            name=obj['ric'],
                        )

                        df = df.append(px_series)
        except Exception as e:
            print(str(e))

        df = df.transpose().sort_index()

        return df

    def _high_corr_report(self, df_input, val_date, ref_price_days, threshold):
        ric_list = list(df_input['ric'].unique())
        start_date = (datetime.strptime(val_date, '%Y-%m-%d') - timedelta(days = ref_price_days)).strftime('%Y-%m-%d')

        batch_size = 100
        batched_ric_list = [ric_list[i:i + batch_size] for i in range(0, len(ric_list), batch_size)]

        qad_price_results = processing_threads(self._qad_price, [(batched_rics, start_date, val_date) for batched_rics in batched_ric_list])

        df_price = pd.concat(qad_price_results, axis=1)

        df_corr = df_price.pct_change(1).corr()

        # ref: https://stackoverflow.com/a/48395484
        df_corr_val = df_corr.mask(np.tril(np.ones(df_corr.shape)).astype(np.bool)).stack().\
            reset_index().rename(columns = {0:'corr'})

        df_corr_val = df_corr_val.merge(df_input.rename(columns = {'UnderlyingBbgYellowKey': 'bbkey_lv0', 'Weight': 'weight_lv0'}),
                                        how = 'left', left_on = 'level_0', right_on = 'ric')

        df_corr_val = df_corr_val.merge(df_input.rename(columns = {'UnderlyingBbgYellowKey': 'bbkey_lv1', 'Weight': 'weight_lv1'}),
                                        how = 'left', left_on = 'level_1', right_on = 'ric')

        df_corr_val = df_corr_val.drop(columns = ['ric_x', 'ric_y'])
        df_corr_val['direction'] = df_corr_val['weight_lv0']*df_corr_val['weight_lv1']
        df_corr_val['direction'] = df_corr_val['direction'].apply(lambda x: 1 if x > 0 else -1)
        df_corr_val['total_weight'] = abs(df_corr_val['weight_lv0']) + abs(df_corr_val['weight_lv1'])

        report = df_corr_val[df_corr_val['corr'] > threshold].sort_values(by = 'total_weight', ascending = False)

        if len(report) <= 0:
            print('no highly correlated pairs in the book ... ')
            return None

        return report

    def _high_corr_grouping(self, port, is_long, val_date, ref_price_days, threshold):
        bbg_rics = list(port['UnderlyingRic'].unique())
        if len(bbg_rics) == 0:
            print('no component in portfolio ... ')
            return pd.DataFrame()

        res = []
        port['ric'] = port['UnderlyingRic']
        df = port[['ric', 'Weight']].drop_duplicates(subset=['ric'])

        df = df[df['Weight'] > 0] if is_long else df[df['Weight'] < 0]

        rpt = self._high_corr_report(df, val_date, ref_price_days, threshold)

        try:
            G = nx.from_pandas_edgelist(rpt, 'level_0', 'level_1')
            groups = list(nx.connected_components(G))
            for group in groups:
                total_weight = df[df['ric'].isin(group)]['Weight'].sum()
                avg_group_corr = rpt[rpt['level_0'].isin(group) | rpt['level_1'].isin(group)]['corr'].mean()
                group_theme = self._security_master_query(list(group), 'ric', ['gic_sector'])
                try:
                    group_theme = group_theme['gic_sector'].mode()[0]
                except:
                    group_theme = ''
                direction = 'long' if total_weight > 0 else 'short'
                item = {'longshort': direction,
                        'group': group,
                        'total_weight': total_weight,
                        'avg_group_corr': avg_group_corr,
                        'group_theme': group_theme}
                res.append(item)
        except:
            pass
        result = pd.DataFrame(res)
        return result

    def _get_polymer_pos(self, val_date, summary = True):
        book = '' if self.book == 'FUND_LEVEL' else self.book

        POSITION_FIELDS = [
            'PositionDate', 'SourceTimestamp', 'Pm', 'SubPm', 'InstrumentType',
            'BbgYellowKey', 'Quantity', 'DeltaUsd', 'IsLong', 'InceptionPnlUsd',
            'Strategy', 'Description', 'InstrumentSubType', 'MarketPrice', 'BookFx',
            'DailyPnlUsd', 'MtdPnlUsd', 'YtdPnlUsd', 'Multiplier', 'IsActive',
            'Custodian', 'GicSector', 'MktCapInMil', 'CountryOfRisk', 'CountryOfExchange', 'underlyingRic'
        ]

        tmp_port = RiskApiDataLoader(token=self.token).get_positions_daily(
            filters={
                "book": book,
                "startDate": val_date,
                "endDate": val_date,
                "isActive": True,
                "onlyEod": True,
                "onlyLatestIntraday": False,
                "columns": POSITION_FIELDS,
                "instrument_types": INSTRUMENT_TYPE_WHITE_LIST,
                "instrument_sub_types": ['Equity', 'Equity Future', 'Equity Option'],
            },
            raw_data=True,
        )

        DATA_COLUMNS_2 = {
            'positionDate': 'Date', 'sourceTimestamp': 'Time', 'pm': 'BUnit', 'subPm': 'Manager', 'instrumentType': 'Type',
            'bbgYellowKey': 'BBKey', 'quantity': 'Quantity', 'deltaUsd': 'MV', 'isLong': 'LS', 'inceptionPnlUsd': 'Cum',
            'strategy': 'Book', 'description': 'Ticker', 'instrumentSubType': 'SubType', 'marketPrice': 'Price', 'bookFx': 'FX',
            'dailyPnlUsd': 'PnL', 'mtdPnlUsd': 'MTD', 'ytdPnlUsd': 'YTD', 'multiplier': 'Multiplier', 'isActive': 'Active',
            'custodian': 'Custodian', 'gicSector': 'Sector', 'mktCapInMil': 'MktCap', 'countryOfRisk': 'Country', 'countryOfExchange': 'ExcCountry',
            'underlyingRic': 'UTicker'
        }

        if len(tmp_port) > 0:
            tmp_port = tmp_port[DATA_COLUMNS_2.keys()].copy()
            tmp_port.rename(columns=DATA_COLUMNS_2, inplace=True)
            tmp_port.Date = pd.to_datetime(tmp_port.Date)
            tmp_port.Time = pd.to_datetime(tmp_port.Time)
            tmp_port.LS = ['Long' if x== True else ('Short' if x == False else 'Flat') for x in tmp_port.LS]
            tmp_port = tmp_port[tmp_port['MV']!= 0 ]

            if self.book == 'FUND_LEVEL':
                tmp_port['BUnit'] = 'FUND_LEVEL'

            if summary:
                res = tmp_port.groupby(['BUnit', 'UTicker'])[['MV']].sum()
                res['GMV'] = abs(res['MV'])
                gmv = res.groupby(['BUnit'])[['GMV']].sum()
                res = res[['MV']].merge(gmv, how='left', left_index=True, right_index=True)
                res['Weight'] = res['MV'] / res['GMV']
                return res
            else:
                return tmp_port
        else:
            print(f'Download_error_valDate: {val_date}')
