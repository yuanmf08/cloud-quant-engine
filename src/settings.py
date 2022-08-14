from datetime import datetime

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from src.utils.factorMapping import (
    BarraFactorNameToExpAse2,
    BarraFactorNameToExpJpe4
)

NOISE_LEVEL = 0.001

PM_EXCLUDING_LIST = ['TREASURY POLYMER', '', 'TREASURY PUBLIC', 'TREASURY_POLYMER', 'xTRADE']
PM_ECM_LIST = ['ECMIPO', 'PO_MAIN', 'IPO_MAIN', 'EDMLO_IPO_MAIN', 'ECMIPO_IPO', 'ECMPO_PO', 'EDMLO_IPO']
PM_HEDGE_LIST = ['xHEDGE', 'XHEDGE']

MAIN_FACTORS = [
    'Specific',
    'RiskIndices',
    'Industry',
    'Country',
    'Currency',
    'World',
    'TradingEffect',
]

BARRA_RT_ADJ = 'Barra Rt Adj'

RISK_INDICES = [
    'Beta',
    'BooktoPrice',
    'DividendYield',
    'EarningsYield',
    'Growth',
    'Leverage',
    'Liquidity',
    'Momentum',
    'NonlinearSize',
    'ResidualVolatility',
    'Size',
]

MAIN_FACTORS_ASE = list(set(['Specific'] + [val['factorType'].replace(" ", "") for key, val in BarraFactorNameToExpAse2.items()]))
RISK_INDICES_ASE = [val['expName'][:-9] for key, val in BarraFactorNameToExpAse2.items() if val['factorType'] == 'Risk Indices']

MAIN_FACTORS_JPE = list(set(['Specific'] + [val['factorType'].replace(" ", "") for key, val in BarraFactorNameToExpJpe4.items()]))
RISK_INDICES_JPE = [val['expName'][:-9] for key, val in BarraFactorNameToExpJpe4.items() if val['factorType'] == 'Risk Indices']

DATE_FORMAT = '%Y-%m-%d'
INCEPTION_DATE = '2019-09-02'
INSTRUMENT_TYPE_WHITE_LIST = ['Equity', 'Future', 'Listed Option', 'OTC Option', 'Index']
OPTION_TYPE_LIST = ['Listed Option', 'OTC Option']
CASH_FX_TYPE_LIST = ['Cash', 'FX Forward']

# if you change this list please also change in front end
INDEX_SUB_TYPE_LIST = ['Exchange Traded Fund', 'Index Option', 'Index Future', 'Equity Basket', 'Index']

DAILY_POSITION_ALL_COLS = {
    'book': {'aggFunc': 'first', 'type': 'category'},
    'pmsKey': {'aggFunc': 'first'},
    'positionDate': {'aggFunc': 'first'},
    'fund': {'aggFunc': 'first', 'type': 'category'},
    'pm': {'aggFunc': 'first', 'type': 'category'},
    'subPm': {'aggFunc': 'first', 'type': 'category'},
    'strategy': {'aggFunc': 'first', 'type': 'category'},
    'instrumentType': {'aggFunc': 'first', 'type': 'category'},
    'instrumentSubType': {'aggFunc': 'first', 'type': 'category'},
    'description': {'aggFunc': 'last'},
    'bbgYellowKey': {'aggFunc': 'last'},
    'underlyingBbgYellowKey': {'aggFunc': 'first'},
    'isActive': {'aggFunc': 'first', 'type': 'category'},
    'quantity': {'aggFunc': 'sum'},
    'multiplier': {'aggFunc': 'first'},
    'localCcy': {'aggFunc': 'first', 'type': 'category'},
    'bookFx': {'aggFunc': 'first'},
    'marketPrice': {'aggFunc': 'first'},
    'avgCostUsd': {'aggFunc': 'first'},
    'notionalUsd': {'aggFunc': 'sum'},
    'marketValueUsd': {'aggFunc': 'sum'},
    'deltaUsd': {'aggFunc': 'sum'},
    'dailyPnlUsd': {'aggFunc': 'sum'},
    'mtdPnlUsd': {'aggFunc': 'sum'},
    'ytdPnlUsd': {'aggFunc': 'sum'},
    'inceptionPnlUsd': {'aggFunc': 'sum'},
    'isLong': {'aggFunc': 'first', 'type': 'category'},
    'countryOfRisk': {'aggFunc': 'first', 'type': 'category'},
    'countryOfExchange': {'aggFunc': 'first', 'type': 'category'},
    'custodian': {'aggFunc': 'first', 'type': 'category'},
    'custodianAccount': {'aggFunc': 'first', 'type': 'category'},
    'trsId': {'aggFunc': 'first', 'type': 'category'},
    'figi': {'aggFunc': 'first', 'type': 'category'},
    'bbgUniqueId': {'aggFunc': 'first'},
    'instrumentId': {'aggFunc': 'first'},
    'underlyingInstrumentId': {'aggFunc': 'first'},
    'gicIndustry': {'aggFunc': 'first', 'type': 'category'},
    'gicSector': {'aggFunc': 'first', 'type': 'category'},
    'avgVol20D': {'aggFunc': 'first'},
    'avgVol30D': {'aggFunc': 'first'},
    'avgVol60D': {'aggFunc': 'first'},
    'avgVol90D': {'aggFunc': 'first'},
    'isin': {'aggFunc': 'first'},
    'cusip': {'aggFunc': 'first'},
    'sedol': {'aggFunc': 'first'},
    'underlyingIsin': {'aggFunc': 'first'},
    'underlyingCusip': {'aggFunc': 'first'},
    'underlyingSedol': {'aggFunc': 'first'},
    'underlyingRic': {'aggFunc': 'first'},
    'barraId': {'aggFunc': 'first'},
    'mktCapInMil': {'aggFunc': 'first'},
}

CURRENCY_COUNTRY_MAPPING = {
    'AUD': 'Australia',
    'CAD': 'Canada',
    'CHF': 'Switzerland',
    'CNH': 'China',
    'CNY': 'China',
    'DKK': 'Denmark',
    'EUR': 'Euro',
    'GBP': 'United Kingdom',
    'HKD': 'Hong Kong',
    'IDR': 'Indonesia',
    'INR': 'India',
    'JPY': 'Japan',
    'KRW': 'South Korea',
    'MYR': 'Malaysia',
    'NOK': 'Norway',
    'NZD': 'New Zealand',
    'PHP': 'Philippines',
    'PLN': 'Poland',
    'SEK': 'Sweden',
    'SGD': 'Singapore',
    'THB': 'Thailand',
    'TWD': 'Taiwan',
    'USD': 'United States',
    'ZAR': 'South Africa',
}

INTERNAL_TRADE_TYPE = [
    'INT',
    'Internal',
    'CROSS',
    'REORG',
]


class UserRoles:
    PLATFORM_STRATEGY = 'platformStrategy'
    ADMIN = 'admin'


class BarraModels:
    GEM3 = 'GEM3 L/S CashMkt'
    JPE4 = 'JPE4 L/S CashMkt'
    ASE2 = 'ASE2 L/S CashMkt'


class PortType:
    BOOK = 'Book'
    HFRX = 'HFRX'
    CUSTOMIZED = 'Customized'


class BaseType:
    GMV = 'Gmv'
    BP = 'Bp'


BookSwitchMapping = [
    {
        'old': 'SNAIT_MAIN',
        'new': 'SHIGN',
        'final': 'SHIGN_FULL',
        'switchStartDate': '2021-10-04'
    },
    {
        'old': 'KWATA_MAIN',
        'new': 'KAZUW',
        'final': 'KAZUW_FULL',
        'switchStartDate': '2021-10-04'
    }
]


def left_merge(
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        left_on=None,
        right_on=None,
        suffixes=('', '_y'),
        left_index=False,
        right_index=False,
):
    if right_on is None:
        right_on = left_on
    return df_left.merge(
        df_right,
        how='left',
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
        left_index=left_index,
        right_index=right_index,
    )


def inner_merge(
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        left_on=None,
        right_on=None,
        suffixes=('', '_y'),
        left_index=False,
        right_index=False,
):
    return df_left.merge(
        df_right,
        how='inner',
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
        left_index=left_index,
        right_index=right_index,
    )


def get_pre_biz_date(
        date_str: str,
        date_format: str
) -> str:
    pre_datetime = datetime.strptime(date_str, date_format) - BDay(1)
    return pre_datetime.strftime(date_format)
