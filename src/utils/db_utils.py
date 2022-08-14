from datetime import date
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.settings import (
    DAILY_POSITION_ALL_COLS,
    get_pre_biz_date,
)
from src.settings import (
    DATE_FORMAT,
    INCEPTION_DATE,
    left_merge,
    BarraModels,
    BookSwitchMapping,
)
from src.utils.db_loader import (
    RiskApiDataLoader,
    get_super_token,
)


def get_benchmark(
        start_date: str,
        end_date: str,
) -> DataFrame:
    """
    get benchmark data from barra | always use fund level data
    """
    risk_api_data_loader = RiskApiDataLoader(token=get_super_token())
    barra_fund_level = risk_api_data_loader.get_barra_pnls(
        from_date=start_date,
        to_date=end_date,
        port='FUND_LEVEL',
        analysis_settings=BarraModels.GEM3,
    )

    benchmark = barra_fund_level[barra_fund_level['factor'] == 'Benchmark']
    benchmark = benchmark[['date', 'dailyReturn']] \
        .rename(columns={'dailyReturn': 'BenchmarkFund'})

    benchmark['date'] = pd.to_datetime(
        benchmark['date'],
        format=DATE_FORMAT,
    ).apply(lambda x: x.strftime(DATE_FORMAT))

    return benchmark


def _process_time_series(
        hist_pnl: DataFrame,
        benchmark: DataFrame = None,
        hist_pnl_no_group: DataFrame = None
) -> DataFrame:
    # check if any pnl not long or short
    hist_pnl['group'] = np.where(
        hist_pnl['group'].isin(['Long', 'Short']),
        hist_pnl['group'],
        'Others'
    )

    # add data for Total
    cols_usd_sum = ['pnl', 'preGmv', 'gmv']
    cols_usd_first = ['preBp', 'bp', 'preAum', 'aum']
    cols_pct = ['returnBp', 'returnGmv', 'returnAum']
    cols_total = cols_usd_sum + cols_usd_first + cols_pct
    res = pd.pivot_table(
        hist_pnl,
        values=cols_total,
        index=['date', 'book'],
        columns=['group'],
        aggfunc='first'
    ).reset_index()

    res.columns = res.columns.map(''.join)

    # if no short or no long add columns
    for nm in ['Long', 'Short', 'Others']:
        for i in cols_total:
            if i + nm not in list(res.columns):
                res[i + nm] = np.NaN

    if hist_pnl_no_group is None:
        for i in cols_usd_sum:
            res[i + 'Total'] = np.where(
                (res[i + 'Long'].isna() & res[i + 'Short'].isna()),
                np.NaN,
                res[i + 'Long'].fillna(0) + res[i + 'Short'].fillna(0)
            )

            if i + 'Others' in res.columns:
                res[i + 'Total'] = np.where(
                    (res[i + 'Total'].isna() & res[i + 'Others'].isna()),
                    res[i + 'Total'],
                    res[i + 'Total'].fillna(0) + res[i + 'Others'].fillna(0)
                )

        for i in cols_usd_first:
            res[i + 'Total'] = res[i + 'Long'].fillna(res[i + 'Short'])
            if i + 'Others' in list(res.columns):
                res[i + 'Total'] = res[i + 'Total'].fillna(res[i + 'Others'])

        # add return on total
        is_na_return = res['returnGmvLong'].isna() & res['returnGmvShort'].isna()
        for i in ['Gmv', 'Aum']:
            res[f'return{i}Total'] = np.where(
                is_na_return,
                np.NaN,
                res['pnlTotal'].div(res[f'pre{i}Total'])
            )

        is_na_bp_return = res['returnBpLong'].isna() & res['returnBpShort'].isna()
        res['returnBpTotal'] = np.where(
            is_na_bp_return,
            np.NaN,
            res['pnlTotal'].div(res['bpTotal'])
        )

    else:
        for i in cols_total:
            hist_pnl_no_group[i + 'Total'] = hist_pnl_no_group[i]
        res = left_merge(
            res,
            hist_pnl_no_group[['date', 'book'] + [i + 'Total' for i in cols_total]],
            ['date', 'book'],
            ['date', 'book'],
        )

    res['utilization'] = res['gmvTotal'].div(res['bpTotal'])
    res.replace([np.inf, -np.inf], np.nan, inplace=True)

    # add benchmark data from barra
    if benchmark is not None:
        res = left_merge(
            res,
            benchmark,
            ['date'],
            ['date'],
        )
    return res


def get_pnl_time_series(
        token: str,
        start_date: str = None,
        end_date: str = None,
        book: str = None,
        include_benchmark: bool = True
) -> DataFrame:
    risk_api_data_loader = RiskApiDataLoader(token=token)
    hist_pnl = risk_api_data_loader.get_pnls(
        start_date=start_date,
        end_date=end_date,
        group_by='longshort',
        book=book,
        include_aum=True,
        include_bp=True
    )
    if len(hist_pnl) == 0:
        return pd.DataFrame()

    # get bp for switch books
    book_switch_list = list(filter(lambda x: x['final'] == book, BookSwitchMapping))
    if book_switch_list and len(book_switch_list) > 0:
        book_switch_details = book_switch_list[0]
        data_period_switch_old = risk_api_data_loader.get_pnls(
            start_date=start_date,
            end_date=end_date,
            group_by='longshort',
            book=book_switch_details['old'],
            include_aum=True,
            include_bp=True
        )

        data_period_switch_new = risk_api_data_loader.get_pnls(
            start_date=start_date,
            end_date=end_date,
            group_by='longshort',
            book=book_switch_details['new'],
            include_aum=True,
            include_bp=True
        )

        data_period_switch_merge = pd.concat([data_period_switch_new, data_period_switch_old])
        data_period_switch_merge.drop_duplicates(subset=['date'], inplace=True)

        # merge bp
        for col in ['bp', 'preBp', 'returnBp', 'cumReturnBp']:
            del hist_pnl[col]
        hist_pnl = left_merge(
            hist_pnl,
            data_period_switch_merge[['date', 'bp']],
            ['date'],
            ['date'],
        )
        hist_pnl['preBp'] = hist_pnl['bp'].shift(1)
        hist_pnl['returnBp'] = hist_pnl['pnl'].div(hist_pnl['bp'])
        hist_pnl['cumReturnBp'] = hist_pnl['returnBp'].add(1).cumprod().subtract(1)

    hist_pnl_no_group = None

    if book == 'FUND_LEVEL':
        hist_pnl_no_group = risk_api_data_loader.get_pnls(
            start_date=start_date,
            end_date=end_date,
            book=book,
            include_aum=True,
            include_bp=True
        )

    benchmark = None
    if include_benchmark:
        benchmark = get_benchmark(start_date=start_date, end_date=end_date)

    return _process_time_series(
        hist_pnl,
        benchmark=benchmark,
        hist_pnl_no_group=hist_pnl_no_group
    )


def get_pnl_time_series_all(
        start_date: str = None,
        end_date: str = None,
        book_group: str = None
) -> DataFrame:
    risk_api_data_loader = RiskApiDataLoader(token=get_super_token())
    hist_pnl = risk_api_data_loader.get_pnls(
        start_date=start_date,
        end_date=end_date,
        group_by='longshort',
        book_group=book_group,
        include_aum=True,
        include_bp=True
    )
    benchmark = get_benchmark(start_date=start_date, end_date=end_date)

    return _process_time_series(hist_pnl, benchmark)


def get_pm_mapping(user_token):
    risk_api_data_loader = RiskApiDataLoader(token=user_token)
    df_book = risk_api_data_loader.get_books()
    return df_book


def get_position_by_date(
        token: str,
        date: str,
        book: str = None,
        columns: List[str] = None
) -> DataFrame:
    cols = list(DAILY_POSITION_ALL_COLS.keys())
    if columns is not None:
        cols = columns

    risk_api_data_loader = RiskApiDataLoader(token=token)
    positions = risk_api_data_loader.get_positions_daily(
        filters={
            'startdate': date,
            'enddate': date,
            'columns': cols,
            'book': book,
        },
    )

    positions['book'] = book

    # modify LTD/YTD/MTD pnl for switched books
    book_switch = [i['final'] for i in BookSwitchMapping]
    if book in book_switch:
        # add pnl to switched books
        book_switch_details = list(filter(lambda x: x['final'] == book, BookSwitchMapping))[0]
        date_switch = book_switch_details['switchStartDate']
        date_switch_pre = get_pre_biz_date(date_switch, DATE_FORMAT)
        if date > date_switch_pre:
            switch_pos = risk_api_data_loader.get_positions_daily(
                filters={
                    'startdate': date_switch_pre,
                    'enddate': date_switch_pre,
                    'columns': cols,
                    'book': book,
                },
            )

            switch_pos['book'] = book
            switch_pos['quantity'] = 0
            switch_pos['deltaUsd'] = 0
            switch_pos['dailyPnlUsd'] = 0
            switch_pos['isActive'] = False


            # combine switch date data:
            if switch_pos is not None:
                if date_switch_pre[:7] != date[:7]:
                    switch_pos['mtdPnlUsd'] = 0
                if date_switch_pre[:4] != date[:4]:
                    switch_pos['ytdPnlUsd'] = 0
                positions = pd.concat([positions, switch_pos])

    return positions


def get_position_by_date_all(date: str) -> DataFrame:
    return get_position_by_date(
        get_super_token(),
        date,
        book='FUND_LEVEL'
    )


def get_barra_return(
        pm: str,
        start_date: str,
        end_date: str,
        barra_model: str = BarraModels.GEM3,
        include_scaled_rt: bool = True
):
    risk_api_data_loader = RiskApiDataLoader(token=get_super_token())
    barra_result = risk_api_data_loader.get_barra_pnls(
        from_date=start_date,
        to_date=end_date,
        port=pm,
        analysis_settings=barra_model,
        include_scaled_rtn=include_scaled_rt
    )

    if barra_model == BarraModels.GEM3:
        rt_col = 'scaledDailyReturn'
    else:
        rt_col = 'dailyReturn'

    if barra_result is not None and len(barra_result) > 0:
        barra_result['pm'] = pm
        barra_result = pd.pivot_table(
            barra_result,
            index=['pm', 'date'],
            columns='factor',
            values=[rt_col],
            aggfunc='first'
        ).reset_index()

        res = barra_result[rt_col].copy()
        res['date'] = barra_result['date']
        res['pm'] = barra_result['pm']

        return res


def get_book_data(params):
    # get all data
    td = date.today().strftime(DATE_FORMAT)
    data_book = get_pnl_time_series(
        token=params.token,
        start_date=INCEPTION_DATE,
        end_date=td,
        book=params.port
    )

    if len(data_book) == 0:
        return [None] * 6

    # check start date and end date for selected port and get data in between
    port_date_list = list(data_book['date'].drop_duplicates())
    port_start_date = max(min(port_date_list), params.startDate)
    port_end_date = min(max(port_date_list), params.endDate)
    data_period = data_book[(data_book['date'] >= port_start_date) & (data_book['date'] <= port_end_date)].copy()

    # get daily snapshot for batting slugging
    date_before_start = max([i for i in port_date_list if i < params.startDate], default=None)
    start_date_data = None
    if date_before_start is not None:
        start_date_data = get_position_by_date(params.token, book=params.port, date=date_before_start)

    date_before_end = max([i for i in port_date_list if i <= params.endDate], default=None)
    end_date_data = get_position_by_date(params.token, book=params.port, date=date_before_end)

    return [data_book, data_period, start_date_data, end_date_data, date_before_start, date_before_end]
