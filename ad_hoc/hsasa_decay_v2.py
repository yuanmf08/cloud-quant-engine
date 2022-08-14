import os.path
from datetime import date, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel
from matplotlib import pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from fastapi.encoders import jsonable_encoder
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mat_ticker
import pickle

from src.utils.db_loader import (
    RiskApiDataLoader,
    auth_check,
    get_token,
)

from src.settings import (
    NOISE_LEVEL,
    left_merge,
    INCEPTION_DATE,
    DATE_FORMAT,
    get_pre_biz_date,
    INSTRUMENT_TYPE_WHITE_LIST,
    BarraModels, INDEX_SUB_TYPE_LIST,
)
from src.utils.db_utils import _process_time_series
from src.utils.threading_utils import processing_threads


def describe(df, stats):
    d = df.describe()
    return d.append(df.reindex(d.columns, axis=1).agg(stats))


class WeightModel(BaseModel):
    min: float
    max: float


class Direction:
    LONG = 'Long'
    SHORT = 'Short'
    ALL = 'ALL'


class BacktestParamsModel(BaseModel):
    num_top: int = 20
    days_past_mkt_rt: int = 20
    de_gross_days: int = 5

    num_name_hedge: int = 5
    hedge_ratio: float = 0.5

    days_to_hedge: int = 7
    new_high_signal_days: int = 3

    steps: int = 2
    stop_loss_pct: float = 0.05
    loss_threshold: float = 0.05
    stop_gain_pct: float = 0.05

    local_new_high_back_days = 60

    high_vol_threshold = 0.5

    # weighting
    weight_de_gross: WeightModel = WeightModel(
        min=0.0,
        max=0.0,
    )
    weight_top_long: WeightModel = WeightModel(
        min=0.3,
        max=0.5,
    )
    weight_top_gain: WeightModel = WeightModel(
        min=0.0,
        max=0.0,
    )
    weight_top_mkt_win: WeightModel = WeightModel(
        min=0.3,
        max=0.5,
    )
    weight_vol: WeightModel = WeightModel(
        min=0.0,
        max=0.0,
    )


class SignalCheckResponse(BaseModel):
    backtestRes: Any
    dailyPnl: Any
    concentration: Any
    heatData: Any


class DailyPosModel(BaseModel):
    date: str
    ric: str
    deltaUsdSod: float
    deltaUsdEod: float
    dailyPnl: float
    tranche: int
    marketPriceSod: float
    marketPriceEod: float
    cumReturn: float
    hedgePrice: float
    steps: float
    highestTranche: int
    currentHighTranche: int
    currentLowTranche: int


class BacktestingDataModel(BaseModel):
    positions: Any
    trades: Any
    pnls: Any
    marketData: Any
    IndexData: Any


class DrawdownHedge:
    def __init__(
            self,
            pm: str,
            pos_start_date: str,
            pos_end_date: str,
            mkt_start_date: str,
            mkt_end_date: str,
            intraday_mode: bool,
    ):
        self._pm = pm
        self._pos_start_date = pos_start_date
        self._pos_end_date = pos_end_date
        self._mkt_start_date = mkt_start_date
        self._mkt_end_date = mkt_end_date
        self._risk_api_data_loader = None

    def _get_trades(self):
        trades = self._risk_api_data_loader.get_trades_filter(
            book=self._pm,
            from_date=self._pos_start_date,
            to_date=self._pos_end_date,
            columns=[
                'pm',
                'counterpartyShortName',
                'dollarNetComm',
                'tradeDate',
                'instrumentType',
                'instrumentSubType',
                'notionalUsd',
                'underlyingRic',
            ],
        )

        return trades

    def _get_positions(self):
        cols = [
            'pm',
            'positionDate',
            'instrumentId',
            'underlyingInstrumentId',
            'deltaUsd',
            'deltaBuy',
            'deltaSell',
            'dailyPnlUsd',
            'quantity',
            'underlyingBbgYellowKey',
            'bbgYellowKey',
            'instrumentSubType',
            'instrumentType',
            'underlyingRic',
            'description',
            'isLong',
        ]

        positions = self._risk_api_data_loader.get_daily_summary_monthly(
            book=self._pm,
            from_date=self._pos_start_date,
            to_date=self._pos_end_date,
            # instrument_types=INSTRUMENT_TYPE_WHITE_LIST,
            columns=cols,
            remove_inactive=False,
        )

        if positions is not None and len(positions) > 0:
            positions = positions[(positions['deltaUsd'].abs() > NOISE_LEVEL) | (positions['dailyPnlUsd'].abs() > NOISE_LEVEL)]
            positions['date'] = positions['positionDate'].str[:10]
            del positions['positionDate']
            positions.sort_values(by='date', inplace=True)

            return positions

    def _get_stock_market_data(self, tk_lst):
        df_mkt = self._risk_api_data_loader.get_qa_stock_quotes(
            rics=tk_lst,
            start_date=self._mkt_start_date,
            end_date=self._mkt_end_date,
        )

        if df_mkt is not None and len(df_mkt) > 0:
            df_mkt = df_mkt.drop_duplicates(subset=['marketDate', 'ric'])
            df_mkt = df_mkt.pivot(index='marketDate', columns='ric', values='returnIndex')
            df_mkt.index = pd.DatetimeIndex(df_mkt.index)
            df_mkt = df_mkt.asfreq('D').ffill()
            df_mkt.index = df_mkt.index.strftime(DATE_FORMAT)

            return df_mkt

    def _get_index_market_data(self, tk_lst):
        api_response = self._risk_api_data_loader.get_qa_index_price(
            rics=tk_lst,
            start_date=self._mkt_start_date,
            end_date=self._mkt_end_date
        )

        if api_response:
            df_mkt = []
            for item in api_response:
                df_mkt_ticker = pd.DataFrame(item['quotes'])
                df_mkt_ticker['ric'] = item['ric']
                df_mkt.append(df_mkt_ticker)
            df_mkt = pd.concat(df_mkt)
            df_mkt['marketDate'] = df_mkt['valueDate']
            df_mkt['marketDate'] = df_mkt['marketDate'].str[:10]

            df_mkt = df_mkt.pivot(index='marketDate', columns='ric', values='returnIndex')
            df_mkt.index = pd.DatetimeIndex(df_mkt.index)
            df_mkt = df_mkt.asfreq('D').ffill()
            df_mkt.index = df_mkt.index.strftime(DATE_FORMAT)

            return df_mkt

    @staticmethod
    def _get_pnls(positions):
        # pnl and drawdown table
        positions['deltaUsdAbs'] = positions['deltaUsd'].abs()
        pos_long = positions[positions['isLong'] == True].copy()
        pos_short = positions[positions['isLong'] == False].copy()

        pnls = pd.DataFrame()
        pnls['dailyPnlUsd'] = positions.groupby(['date'])['dailyPnlUsd'].sum()
        pnls['dailyPnlUsdLong'] = pos_long.groupby(['date'])['dailyPnlUsd'].sum()
        pnls['dailyPnlUsdShort'] = pos_short.groupby(['date'])['dailyPnlUsd'].sum()
        pnls['gmv'] = positions.groupby(['date'])['deltaUsdAbs'].sum()
        pnls['gmvLong'] = pos_long.groupby(['date'])['deltaUsdAbs'].sum()
        pnls['gmvShort'] = pos_short.groupby(['date'])['deltaUsdAbs'].sum()
        pnls['returnGmv'] = pnls['dailyPnlUsd'].div(pnls['gmv'].shift(1)).fillna(0).replace([np.inf, -np.inf], 0)
        pnls['returnGmvLong'] = pnls['dailyPnlUsdLong'].div(pnls['gmvLong'].shift(1)).fillna(0).replace([np.inf, -np.inf], 0)
        pnls['returnGmvShort'] = pnls['dailyPnlUsdShort'].div(pnls['gmvShort'].shift(1)).fillna(0).replace([np.inf, -np.inf], 0)
        pnls['cumPnl'] = pnls['dailyPnlUsd'].cumsum()
        pnls['cumPnlLong'] = pnls['dailyPnlUsdLong'].cumsum()
        pnls['cumPnlShort'] = pnls['dailyPnlUsdShort'].cumsum()
        pnls['cumReturnGmv'] = pnls['returnGmv'].add(1).cumprod().subtract(1)
        pnls['cumReturnGmvLong'] = pnls['returnGmvLong'].add(1).cumprod().subtract(1)
        pnls['cumReturnGmvShort'] = pnls['returnGmvShort'].add(1).cumprod().subtract(1)

        return pnls

    @staticmethod
    def _score_top_exp(
            params: BacktestParamsModel,
            current_pos_by_ric,
            direction: Direction = Direction.LONG,
    ):
        weight_top_long = params.weight_top_long
        num_top = params.num_top

        # check top holdings
        if direction == Direction.LONG:
            gmv_top = current_pos_by_ric[current_pos_by_ric['deltaUsd'] > 0].sort_values(by=['deltaUsd'], ascending=False).head(num_top)
        elif direction == Direction.SHORT:
            gmv_top = current_pos_by_ric[current_pos_by_ric['deltaUsd'] < 0].sort_values(by=['deltaUsd']).head(num_top)
        else:
            gmv_top = current_pos_by_ric.sort_values(by=['deltaUsdAbs'], ascending=False).head(num_top)

        gmv_top['scoreGmv'] = np.linspace(weight_top_long.max, weight_top_long.min, num=num_top, endpoint=True)[:len(gmv_top)]

        return gmv_top[['scoreGmv']]

    @staticmethod
    def _score_top_pnl_gain(
            params: BacktestParamsModel,
            hist_positions,
            check_ticker_list
    ):
        weight_top_gain = params.weight_top_gain
        num_top = params.num_top

        # check top pnl gain
        pnl_top_gain = pd.DataFrame()
        pnl_top_gain['dailyPnlUsd'] = hist_positions.groupby(['underlyingRic'])['dailyPnlUsd'].sum()
        pnl_top_gain = pnl_top_gain[pnl_top_gain.index.isin(check_ticker_list)]
        pnl_top_gain = pnl_top_gain.sort_values(by=['dailyPnlUsd'], ascending=False).head(num_top)
        pnl_top_gain['scorePnlGain'] = np.linspace(weight_top_gain.max, weight_top_gain.min, num=num_top, endpoint=True)[:len(pnl_top_gain)]

        return pnl_top_gain

    @staticmethod
    def _score_de_gross(
            params: BacktestParamsModel,
            trades,
            hist_positions,
            current_pos_by_ric,
            check_ticker_list,
    ):
        weight_de_gross = params.weight_de_gross
        num_top = params.num_top
        de_gross_days = params.de_gross_days

        # 1 check de gross signal
        date_list = list(hist_positions['date'].drop_duplicates().sort_values())
        check_date_list = date_list[-de_gross_days:]

        ed_dt = check_date_list[-1]
        stt_dt = check_date_list[0]
        trade_filtered = trades[
            (trades['tradeDate'] <= ed_dt) &
            (trades['tradeDate'] >= stt_dt) &
            trades['underlyingRic'].isin(check_ticker_list)].copy()
        trade_by_ric = pd.DataFrame()
        trade_by_ric['notionalUsd'] = trade_filtered.groupby(['underlyingRic'])['notionalUsd'].sum()

        trade_by_ric = left_merge(
            trade_by_ric,
            current_pos_by_ric['deltaUsd'],
            left_index=True,
            right_index=True,
        )

        trade_by_ric['deGrossUsd'] = trade_by_ric['notionalUsd'] * trade_by_ric['deltaUsd'].div(trade_by_ric['deltaUsd'].abs())
        trade_by_ric = trade_by_ric.sort_values(by='deGrossUsd').head(num_top)
        trade_by_ric['scoreDeGross'] = np.linspace(weight_de_gross.max, weight_de_gross.min, num=num_top, endpoint=True)[:len(trade_by_ric)]

        return trade_by_ric[['scoreDeGross', 'deGrossUsd']]

    @staticmethod
    def _score_mkt_rt(
            params: BacktestParamsModel,
            mkt_data_trans,
            hist_positions,
            current_pos_by_ric,
            check_ticker_list,
    ):
        col_rt_signal = 'rtSignal'
        days_past_mkt_rt = params.days_past_mkt_rt
        weight_top_mkt_win = params.weight_top_mkt_win
        num_top = params.num_top

        date_list = list(hist_positions['date'].drop_duplicates().sort_values())
        check_date_list = date_list[-days_past_mkt_rt:]

        # check top market returns
        stt_dt = check_date_list[0]
        ed_dt = check_date_list[-1]

        mkt_rt_past_days = mkt_data_trans[mkt_data_trans.index.isin(check_ticker_list)].copy()
        mkt_rt_past_days[col_rt_signal] = mkt_rt_past_days[ed_dt] / mkt_rt_past_days[stt_dt] - 1
        mkt_rt_past_days = left_merge(
            mkt_rt_past_days,
            current_pos_by_ric['deltaUsd'],
            left_index=True,
            right_index=True,
        )
        mkt_rt_past_days[col_rt_signal] = (mkt_rt_past_days[col_rt_signal] * mkt_rt_past_days['deltaUsd']).div(mkt_rt_past_days['deltaUsd'].abs())

        mkt_rt_top = mkt_rt_past_days.sort_values(by=[col_rt_signal], ascending=False).head(num_top)
        mkt_rt_top['scoreMktRt'] = np.linspace(weight_top_mkt_win.max, weight_top_mkt_win.min, num=num_top, endpoint=True)[:len(mkt_rt_top)]

        return mkt_rt_top[[col_rt_signal, 'scoreMktRt']]

    @staticmethod
    def _score_vol(
            params: BacktestParamsModel,
            mkt_data,
            hist_positions,
    ):
        days_past_mkt_rt = params.days_past_mkt_rt
        high_vol_threshold = params.high_vol_threshold
        date_list = list(hist_positions['date'].drop_duplicates().sort_values())
        check_date_list = date_list[-days_past_mkt_rt:]

        # check top market returns
        stt_dt = check_date_list[0]
        ed_dt = check_date_list[-1]

        rt_data = mkt_data.pct_change()
        rt_data = rt_data[(rt_data.index >= stt_dt) & (rt_data.index <= ed_dt)]
        rt_data_t = rt_data.T
        rt_data_t['vol'] = rt_data_t.std(axis=1) * (252 ** 0.5)

        rt_data_t['scoreVol'] = np.where(
            rt_data_t['vol'] > high_vol_threshold,
            -1,
            0,
        )

        return rt_data_t[['scoreVol', 'vol']]

    def gen_hedge_trade_score(
            self,
            backtesting_data: BacktestingDataModel,
            params: BacktestParamsModel,
            position_date,
            direction: Direction = Direction.LONG
    ):
        mkt_data = backtesting_data.marketData
        positions = backtesting_data.positions
        trades = backtesting_data.trades

        mkt_data_trans = mkt_data.T
        mkt_data_trans.columns = [str(i)[:10] for i in list(mkt_data_trans.columns)]

        current_position = positions[positions['date'] == position_date].copy()
        if len(current_position) == 0:
            return

        hist_positions = positions[positions['date'] <= position_date].copy()

        # group by ticker
        current_pos_by_ric = pd.DataFrame()
        current_pos_by_ric['deltaUsd'] = current_position.groupby(['underlyingRic'])['deltaUsd'].sum()
        current_pos_by_ric['deltaUsdAbs'] = current_position['deltaUsd'].abs()
        current_pos_by_ric['date'] = position_date

        # add market price
        mkt_data_trans['marketPrice'] = mkt_data_trans[position_date]
        current_pos_by_ric = left_merge(
            current_pos_by_ric,
            mkt_data_trans[['marketPrice']],
            left_index=True,
            right_index=True
        )

        # still exist tickers
        check_ticker_list = list(current_pos_by_ric[current_pos_by_ric['deltaUsd'].abs() > NOISE_LEVEL].index.drop_duplicates())

        # check top exp
        gmv_top = self._score_top_exp(params, current_pos_by_ric, direction=direction)

        # check top de gross
        trade_top_short = self._score_de_gross(params, trades, hist_positions, current_pos_by_ric, check_ticker_list)

        pnl_top_gain = self._score_top_pnl_gain(params, hist_positions, check_ticker_list)

        mkt_rt_top = self._score_mkt_rt(params, mkt_data_trans, hist_positions, current_pos_by_ric, check_ticker_list)

        # check vol
        vol_past_days = self._score_vol(params, mkt_data, hist_positions)


        """ combine together """
        for df in [gmv_top, trade_top_short, pnl_top_gain, mkt_rt_top, vol_past_days]:
            current_pos_by_ric = left_merge(current_pos_by_ric, df, left_index=True, right_index=True)

        score_col_list = [i for i in list(current_pos_by_ric.columns) if i.startswith('score')]
        current_pos_by_ric['score'] = current_pos_by_ric[score_col_list].fillna(0).sum(axis=1)

        return current_pos_by_ric

    def _backtesting(
            self,
            backtesting_data: BacktestingDataModel,
            params: BacktestParamsModel,
            direction: Direction = Direction.LONG,
            signal_col='cumReturnGmv',
    ):
        num_name_hedge = params.num_name_hedge
        hedge_ratio = params.hedge_ratio

        days_to_hedge = params.days_to_hedge

        new_high_signal_days = params.new_high_signal_days

        steps = params.steps
        loss_threshold = params.loss_threshold
        stop_loss_pct = params.stop_loss_pct
        stop_gain_pct = params.stop_gain_pct
        local_new_high_back_days = params.local_new_high_back_days

        pnls = backtesting_data.pnls
        mkt_data = backtesting_data.marketData

        mkt_data_trans = mkt_data.T
        hedge_count = 0
        is_trading = False
        days_after_new_high = 0

        pnls['maxDrawdown'] = pnls[signal_col] - pnls[signal_col].rolling(local_new_high_back_days).max()
        pnls['date'] = pnls.index

        current_hedge_dict: Dict[str, DailyPosModel] = dict()
        df_result = []

        for row in pnls.itertuples():
            current_date = getattr(row, 'date')
            max_drawdown = getattr(row, 'maxDrawdown')
            mkt_dict = dict(zip(list(mkt_data_trans.index), list(mkt_data_trans[current_date])))

            # check new high
            new_high = max_drawdown > -0.000001

            if new_high:
                days_after_new_high += 1
            else:
                days_after_new_high = 0

            if days_after_new_high >= new_high_signal_days:
                trade_signal = True
            else:
                trade_signal = False

            if is_trading:
                # print(f'Is Trading: {current_date}, {trade_signal}')

                # backtesting
                for key, val in current_hedge_dict.items():
                    val.date = current_date

                    # create sod
                    val.marketPriceSod = val.marketPriceEod
                    val.deltaUsdSod = val.deltaUsdEod

                    val.marketPriceEod = mkt_dict.get(val.ric) or np.nan
                    val.deltaUsdEod = val.marketPriceEod / val.marketPriceSod * val.deltaUsdSod

                    val.dailyPnl = val.deltaUsdEod - val.deltaUsdSod

                    # EOD decisions
                    if val.tranche > 0:
                        new_tranche = val.tranche
                        cum_rt = (val.marketPriceEod / val.hedgePrice - 1) * val.deltaUsdEod / abs(val.deltaUsdEod)
                        tranche_stop_loss = -stop_loss_pct - stop_loss_pct / steps * val.currentLowTranche
                        tranche_threshold = -stop_loss_pct / steps * val.tranche
                        tranche_stop_gain = stop_gain_pct * val.tranche

                        # add tranche
                        if cum_rt < tranche_threshold and val.tranche < steps and val.tranche == val.currentHighTranche:
                            new_tranche = val.tranche + 1
                            val.currentHighTranche = val.currentHighTranche + 1

                        elif cum_rt < tranche_stop_loss:
                            # stop loss
                            new_tranche = val.tranche - 1
                            val.currentLowTranche = val.currentLowTranche + 1

                        elif cum_rt > tranche_stop_gain:
                            # stop gain
                            new_tranche = val.tranche - 1
                            val.currentHighTranche = val.currentHighTranche - 1

                        val.deltaUsdEod = val.deltaUsdEod / val.tranche * new_tranche
                        val.tranche = new_tranche
                        # val.highestTranche = max(new_tranche, val.highestTranche)

                df_result.append(pd.DataFrame(jsonable_encoder(list(current_hedge_dict.values()))))

                # check
                hedge_count += 1
                if hedge_count > days_to_hedge:
                    hedge_count = 0
                    is_trading = False

            elif trade_signal:
                # print(f'Hedge date: {current_date}, {trade_signal}')
                # enter trade
                score = self.gen_hedge_trade_score(
                    backtesting_data,
                    params=params,
                    position_date=current_date,
                    direction=direction,
                )
                score.sort_values(['score'], ascending=False, inplace=True)

                # only long or short
                if direction == Direction.LONG:
                    score = score[score['deltaUsd'] > 0]
                elif direction == Direction.SHORT:
                    score = score[score['deltaUsd'] < 0]

                names_to_hedge = score.head(num_name_hedge)

                current_hedge_dict: Dict[str, DailyPosModel] = dict()
                for row_data in names_to_hedge.itertuples():
                    ric = getattr(row_data, 'Index')
                    pc = mkt_dict.get(ric)
                    if pc is None:
                        pc = np.nan
                    current_hedge_dict[ric] = DailyPosModel(
                        ric=ric,
                        date=getattr(row_data, 'date'),
                        deltaUsdSod=-getattr(row_data, 'deltaUsd') / steps * hedge_ratio,
                        deltaUsdEod=-getattr(row_data, 'deltaUsd') / steps * hedge_ratio,
                        dailyPnl=0,
                        tranche=1,
                        marketPriceSod=pc,
                        marketPriceEod=pc,
                        steps=steps,
                        cumReturn=0,
                        hedgePrice=pc,
                        highestTranche=1,
                        currentHighTranche=1,
                        currentLowTranche=1,
                    )

                is_trading = True

            else:
                hedge_count = 0

        if len(df_result) > 0:
            backtesting = pd.concat(df_result)
        else:
            backtesting = pd.DataFrame()
        # backtesting.to_excel(r'C:\Users\minfengy\Downloads\klkl.xlsx', index=False)

        return backtesting

    def profile_concentration(
            self,
            positions,
            pdf=None,
            fig_size=(11.69, 8.27),
            grid_title='Profile: Concentration'
    ):
        # group position by underlying and date
        agg_func = {
            'date': 'first',
            'underlyingInstrumentId': 'first',
            'deltaUsd': 'sum',
            'dailyPnlUsd': 'sum',
            'underlyingBbgYellowKey': 'first',
            'instrumentSubType': 'first',
            'instrumentType': 'first',
        }

        pos_by_id_date = positions[list(agg_func.keys())].groupby(['date', 'underlyingBbgYellowKey'], observed=True).agg(agg_func).reset_index(drop=True)
        pos_by_id_date.sort_values(by='date', inplace=True)
        pos_by_id_date['deltaUsdAbs'] = pos_by_id_date['deltaUsd'].abs()
        pos_by_id_date['deltaUsdAbs'] = pos_by_id_date['deltaUsd'].abs()
        pos_by_id_date['rolling20DPnl'] = pos_by_id_date.groupby('underlyingBbgYellowKey')['dailyPnlUsd'].transform(lambda s: s.rolling(20, min_periods=1).sum())

        con_by_date = pd.DataFrame()
        con_by_date['deltaUsdAbs'] = pos_by_id_date.groupby(['date'], observed=True)['deltaUsdAbs'].sum()
        con_by_date['dailyPnlUsd'] = pos_by_id_date.groupby(['date'], observed=True)['dailyPnlUsd'].sum()
        con_by_date['rolling20DPnlPos'] = pos_by_id_date[pos_by_id_date['rolling20DPnl'] > 0].groupby(['date'], observed=True)['rolling20DPnl'].sum()
        con_by_date['rolling20DPnlNeg'] = pos_by_id_date[pos_by_id_date['rolling20DPnl'] < 0].groupby(['date'], observed=True)['rolling20DPnl'].sum()

        # cal equity only concentration
        pos_by_id_equity = pos_by_id_date[(~pos_by_id_date['instrumentSubType'].isin(INDEX_SUB_TYPE_LIST)) & pos_by_id_date['instrumentType'].isin(INSTRUMENT_TYPE_WHITE_LIST)]
        pos_by_id_equity = pos_by_id_equity[pos_by_id_equity['underlyingBbgYellowKey'] != '']

        # # get cum pnl by name for rolling 20 days
        # pos_by_id_equity['rolling20DPnl'] = pos_by_id_equity.groupby('underlyingBbgYellowKey')['dailyPnlUsd'].transform(lambda s:s.rolling(20, min_periods=1))

        # get concen by date
        cons_days = [1, 5, 10]
        for n in cons_days:
            pos_by_id_equity_long = pos_by_id_equity[pos_by_id_equity['deltaUsd'] > 0].copy()
            con_by_date[f'top{n}GmvPctLong'] = pos_by_id_equity_long.groupby('date', observed=True)['deltaUsd'].apply(lambda x, m=n: x.nlargest(m).sum()).div(con_by_date['deltaUsdAbs'])
            con_by_date[f'top{n}PnlPctLong'] = pos_by_id_equity.groupby('date', observed=True).apply(lambda x, m=n: x[x['rolling20DPnl'] > 0].nlargest(m, 'rolling20DPnl')['rolling20DPnl'].sum()).div(con_by_date['rolling20DPnlPos'])
            pos_by_id_equity_short = pos_by_id_equity[pos_by_id_equity['deltaUsd'] < 0].copy()
            con_by_date[f'top{n}GmvPctShort'] = pos_by_id_equity_short.groupby('date', observed=True)['deltaUsd'].apply(lambda x, m=n: x.nsmallest(m).sum()).div(con_by_date['deltaUsdAbs'])
            con_by_date[f'top{n}PnlPctShort'] = pos_by_id_equity.groupby('date', observed=True).apply(lambda x, m=n: x[x['rolling20DPnl'] < 0].nsmallest(m, 'rolling20DPnl')['rolling20DPnl'].sum()).div(con_by_date['rolling20DPnlNeg'])

        con_by_date['pm'] = self._pm

        # pnl concentration
        d = 20

        col_delta_list = {f'top{i}GmvPct{j}': f'Top {i} {j}' for i in cons_days for j in ['Long', 'Short']}
        col_pnl_list = {f'top{i}PnlPct{j}': f'Top {i} {j}' for i in cons_days for j in ['Long', 'Short']}
        cons_delta_plot = con_by_date[list(col_delta_list.keys())].copy()
        cons_delta_plot.columns = list(col_delta_list.values())
        cons_pnl_plot = con_by_date[list(col_pnl_list.keys())].copy()
        cons_pnl_plot.columns = list(col_pnl_list.values())

        data_list = {
            'GMV Concentration': cons_delta_plot,
            'PNL Concentration': cons_pnl_plot
        }

        # plot
        col = 2
        row = 1
        fig, axes = plt.subplots(col, row, figsize=fig_size)
        # plt.xticks(rotation=45)

        fig.suptitle(grid_title)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        i = 0
        for nm, dt in data_list.items():
            line = dt.plot.line(ax=axes[i])
            line.set_title(nm)
            plt.xticks(rotation=90)
            # line.yaxis.set_major_formatter(mat_ticker.FuncFormatter('{0:.0%}'.format))

            # dt['date'] = dt.index
            # g = sns.lineplot(data=dt, x='date', ax=axes[i // row, i % row])
            # g.set(title=nm)
            i += 1

        for ax in axes.flat:
            ax.yaxis.set_major_formatter(mat_ticker.FuncFormatter('{0:.0%}'.format))
            ax.tick_params(axis='x', rotation=45)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            plt.show()


        return con_by_date

    def profile_volatility(
            self,
            positions,
            market_data,
            pdf=None,
            fig_size=(11.69, 8.27),
            grid_title='Profile: Volatility',
    ):
        vol_list = [20, 60, 90]
        market_rt = market_data.pct_change()
        hist_vol = pd.DataFrame()
        hist_vol.index = market_rt.unstack().index
        for d in vol_list:
            hist_vol[f'vol{d}D'] = (market_rt.rolling(d).std() * (252 ** 0.5)).unstack()
        hist_vol.reset_index(inplace=True)
        hist_vol['date'] = hist_vol['marketDate']
        del hist_vol['marketDate']

        # group position by underlying and date
        agg_func = {
            'date': 'first',
            'deltaUsd': 'sum',
            'underlyingRic': 'first',
            'instrumentSubType': 'first',
            'instrumentType': 'first',
        }
        pos_by_id_date = positions[list(agg_func.keys())].groupby(['date', 'underlyingRic'], observed=True).agg(agg_func).reset_index(drop=True)
        pos_by_id_date.sort_values(by='date', inplace=True)
        pos_by_id_date['deltaUsdAbs'] = pos_by_id_date['deltaUsd'].abs()
        pos_by_id_date = left_merge(
            pos_by_id_date,
            hist_vol,
            ['date', 'underlyingRic'],
            ['date', 'ric']
        )

        vol_by_date = pd.DataFrame()
        vol_by_date['deltaUsdAbs'] = pos_by_id_date.groupby(['date'], observed=True)['deltaUsdAbs'].sum()

        # cal equity only concentration
        pos_by_id_equity = pos_by_id_date[(~pos_by_id_date['instrumentSubType'].isin(INDEX_SUB_TYPE_LIST)) & pos_by_id_date['instrumentType'].isin(INSTRUMENT_TYPE_WHITE_LIST)]
        pos_by_id_equity = pos_by_id_equity[pos_by_id_equity['underlyingRic'] != '']

        # get concen by date
        for n in vol_list:
            pos_by_id_equity_long = pos_by_id_equity[pos_by_id_equity['deltaUsd'] > 0].copy()
            vol_by_date[f'weightedVol{n}DLong'] = pos_by_id_equity_long.groupby('date', observed=True).apply(lambda x, m=n: (x[~x[f'vol{m}D'].isna()][f'vol{m}D'] * x['deltaUsdAbs']).sum() / (x[~x[f'vol{m}D'].isna()]['deltaUsdAbs'].sum()))
            pos_by_id_equity_short = pos_by_id_equity[pos_by_id_equity['deltaUsd'] < 0].copy()
            vol_by_date[f'weightedVol{n}DShort'] = pos_by_id_equity_short.groupby('date', observed=True).apply(lambda x, m=n: (x[~x[f'vol{m}D'].isna()][f'vol{m}D'] * x['deltaUsdAbs']).sum() / (x[~x[f'vol{m}D'].isna()]['deltaUsdAbs'].sum()))

        vol_by_date['pm'] = self._pm

        col_list = [f'weightedVol{i}D{j}' for i in vol_list for j in ['Long', 'Short']]
        vols_plot = vol_by_date[col_list].copy()

        lines = vols_plot.plot.line(figsize=fig_size)
        plt.title(grid_title)
        plt.xticks(rotation=45)
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)

        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

        return vol_by_date

    @staticmethod
    def profile_drawdown(
            pnls,
            pdf=None,
            fig_size=(11.69, 8.27),
            grid_title='Profile: Drawdown',
    ):
        new_high_days_list = range(1, 10)
        drawdown_check_days = 5

        local_new_high_days = 30

        signal_col = 'cumReturnGmv'

        pnls['maxDrawdown'] = pnls[signal_col] - pnls[signal_col].rolling(local_new_high_days).max()
        pnls['newHigh'] = np.where(pnls['maxDrawdown'] > -0.000001, 1, 0)
        g = pnls['newHigh'].eq(0).cumsum()
        pnls['cumNewHighDays'] = pnls['newHigh'].groupby(g).transform('cumsum')

        col_return_in_days = f'returnIn{drawdown_check_days}Days'
        col_max_drawdown_in_days = f'maxDrawdownIn{drawdown_check_days}Days'
        col_max_high_in_days = f'maxHighIn{drawdown_check_days}Days'
        pnls[col_return_in_days] = (pnls[signal_col].shift(-drawdown_check_days) + 1).div(pnls[signal_col] + 1) - 1
        pnls[col_max_drawdown_in_days] = (pnls[signal_col].rolling(drawdown_check_days).min().shift(-drawdown_check_days) + 1).div(pnls[signal_col] + 1) - 1
        pnls[col_max_high_in_days] = (pnls[signal_col].rolling(drawdown_check_days).max().shift(-drawdown_check_days) + 1).div(pnls[signal_col] + 1) - 1

        # plot
        col = 2
        row = 2
        fig, axes = plt.subplots(col, row, figsize=fig_size)

        fig.suptitle(grid_title)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        graph1 = sns.boxplot(
            x='cumNewHighDays',
            y=col_max_drawdown_in_days,
            data=pnls[pnls['cumNewHighDays'] > 0],
            palette="Blues",
            ax=axes[0, 0]
        )
        graph1.set_title(f'Max Drawdown in {drawdown_check_days} Days')
        graph1.axhline(0, linestyle='--', linewidth=1)

        graph2 = sns.boxplot(
            x='cumNewHighDays',
            y=col_return_in_days,
            data=pnls[pnls['cumNewHighDays'] > 0],
            palette="Blues",
            ax=axes[0, 1]
        )
        graph2.set_title(f'Return in {drawdown_check_days} Days')
        graph2.axhline(0, linestyle='--', linewidth=1)

        graph3 = sns.boxplot(
            x='cumNewHighDays',
            y=col_max_high_in_days,
            data=pnls[pnls['cumNewHighDays'] > 0],
            palette="Blues",
            ax=axes[1, 0]
        )
        graph3.set_title(f'High in {drawdown_check_days} Days')
        graph3.axhline(0, linestyle='--', linewidth=1)

        for ax in axes.flat:
            ax.yaxis.set_major_formatter(mat_ticker.FuncFormatter('{0:.1%}'.format))

        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            plt.show()


    def multi_params_backtesting(
            self,
            backtesting_data: BacktestingDataModel,
            default_params: BacktestParamsModel,
            shock_params: dict,
            direction: Direction = Direction.LONG,
            signal_col='cumReturnGmv'
    ):
        params = default_params.copy()
        backtest_res_list = []
        daily_pnl_list = []

        override_dict = {}
        for key, val in shock_params.items():
            key_to_show = ''.join([s.capitalize() for s in key.split('_')])
            override_dict_len = len(override_dict)
            if override_dict_len > 0:
                new_override_dict = {}
                for over_key, over_val in override_dict.items():
                    for i in val:
                        new_override_dict[f'{over_key}-{key_to_show}_{i}'] = over_val.copy()
                        new_override_dict[f'{over_key}-{key_to_show}_{i}'][key] = i
                override_dict = new_override_dict

            else:
                for i in val:
                    override_dict[f'{key_to_show}_{i}'] = {key: i}

        for name, overrides in override_dict.items():
            for key, val in overrides.items():
                setattr(params, key, val)
            backtest_res = self._backtesting(
                backtesting_data,
                params,
                direction=direction,
                signal_col=signal_col,
            )
            backtest_res_list.append(backtest_res)

            # daily pnl
            if len(backtest_res) > 0:
                daily = pd.DataFrame()
                daily['dailyPnl'] = backtest_res.groupby(['date'])['dailyPnl'].sum()
                backtest_res['deltaUsdSodAbs'] = backtest_res['deltaUsdSod'].abs()
                daily['gmvSod'] = backtest_res.groupby(['date'])['deltaUsdSodAbs'].sum()
                daily['returnGmvHedge'] = daily['dailyPnl'].div(daily['gmvSod'])
                for key, val in overrides.items():
                    daily[key] = val
                daily_pnl_list.append(daily)

        if len(daily_pnl_list) > 0:
            daily_pnl_list = pd.concat(daily_pnl_list)
            daily_pnl_list['cumPnl'] = daily_pnl_list['dailyPnl'].cumsum()
            daily_pnl_list['cumReturnGmvHedge'] = (daily_pnl_list['returnGmvHedge'] + 1).cumprod() - 1
            # d = describe(final_res, ['sum', 'skew', 'mad', 'kurt'])

            backtest_res_list = pd.concat(backtest_res_list)

        return backtest_res_list, daily_pnl_list

    def get_data(self) -> BacktestingDataModel:
        pos = self._get_positions()
        if pos is None:
            return
        ticker_list = list(pos['underlyingRic'].drop_duplicates().dropna())
        mkt_dat = self._get_stock_market_data(ticker_list)

        # try get index data
        index_list = [i for i in ticker_list if i not in list(mkt_dat.columns)]
        index_dat = self._get_index_market_data(index_list)

        mkt_dat = pd.concat([mkt_dat, index_dat], axis=1)

        tds = self._get_trades()
        pnls = self._get_pnls(pos)

        return BacktestingDataModel(
            positions=pos,
            marketData=mkt_dat,
            trades=tds,
            pnls=pnls,
            IndexData=index_dat,
        )

    def run_backtest_by_step(
            self,
            backtesting_data: BacktestingDataModel,
            days_to_hedge,
            new_high_signal_days,
            default_param: BacktestParamsModel = None,
            direction: Direction = Direction.LONG,
            signal_col='cumReturnGmv'
    ):
        if default_param is None:
            default_param = BacktestParamsModel()

        default_param.days_to_hedge = days_to_hedge
        default_param.new_high_signal_days = new_high_signal_days

        test_params = {
            'steps': range(1, 4),
            # 'hedge_ratio': range(1, 10),
        }

        backtest_res, daily_pnl = self.multi_params_backtesting(
            backtesting_data,
            default_params=default_param,
            shock_params=test_params,
            direction=direction,
            signal_col=signal_col,
        )

        res_chart_data = pd.pivot_table(daily_pnl, values='dailyPnl', index=['date'], columns=['steps'], aggfunc=np.sum)
        res_chart_data.cumsum().plot()
        plt.show()

        return backtest_res, daily_pnl

    # def performance_metrics(self, daily_pnl):
    #
    #

    def signal_check(
            self,
            backtesting_data: BacktestingDataModel,
            default_param: BacktestParamsModel = None,
            direction: Direction = Direction.LONG,
            signal_col='cumReturnGmv'
    ) -> SignalCheckResponse:
        if default_param is None:
            default_param = BacktestParamsModel()

        days_to_hedge_range = range(1, 10)
        new_high_signal_days_range = range(1, 10)

        shock_params = {
            'days_to_hedge': days_to_hedge_range,
            'new_high_signal_days': new_high_signal_days_range,
        }

        positions = backtesting_data.positions
        pnls = backtesting_data.pnls.copy()
        pnls

        if positions is None:
            return

        backtest_res, daily_pnl = self.multi_params_backtesting(
            backtesting_data,
            default_params=default_param,
            shock_params=shock_params,
            direction=direction,
            signal_col=signal_col,
        )

        if len(daily_pnl) > 0:
            heatdata = pd.pivot_table(
                daily_pnl,
                values='dailyPnl',
                index=['days_to_hedge'],
                columns=['new_high_signal_days'],
                aggfunc=np.sum
            )

            result = SignalCheckResponse(
                backtestRes=backtest_res,
                dailyPnl=daily_pnl,
                heatData=heatdata,
            )

            return result

    def signal_utility_func(
            self,
            pnl_heat_data,
            max_drawdown_heat_data,
            vol_heat_data,
            cum_rt_heat_data,
            sharpe_heat_data,
    ):


    @staticmethod
    def perf_metrix_heatmap_plot(
            daily_pnl,
            col1,
            col2,
            pdf=None,
            fig_size=(11.69, 8.27),
            grid_title=''
    ):
        pnl_heat_data = pd.pivot_table(
            daily_pnl,
            values='dailyPnl',
            index=[col1],
            columns=[col2],
            aggfunc=np.sum
        )

        max_drawdown_heat_data = pd.pivot_table(
            daily_pnl,
            values='dailyPnl',
            index=[col1],
            columns=[col2],
            aggfunc=lambda df: (df - df.cummax()).min(),
        )

        vol_heat_data = pd.pivot_table(
            daily_pnl,
            values='returnGmvHedge',
            index=[col1],
            columns=[col2],
            aggfunc=lambda df: df.std() * (252 ** 0.5),
        )

        cum_rt_heat_data = pd.pivot_table(
            daily_pnl,
            values='returnGmvHedge',
            index=[col1],
            columns=[col2],
            aggfunc=lambda df: (df + 1).prod() - 1,
        )

        sharpe_heat_data = cum_rt_heat_data / vol_heat_data

        heat_data_list = {
            'cumPnl': pnl_heat_data,
            'maxDrawdown': max_drawdown_heat_data,
            'Volatility': vol_heat_data,
            'cumRt': cum_rt_heat_data,
            'sharpe': sharpe_heat_data
        }

        # plot heat map
        col = 3
        row = 2
        fig, axes = plt.subplots(col, row, figsize=fig_size)
        fig.suptitle(grid_title)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        i = 0
        for nm, dt in heat_data_list.items():
            sns.heatmap(dt, cmap="YlGnBu", ax=axes[i // row, i % row]).set(title=nm)
            i += 1

        # sns.heatmap(pnl_heat_data, cmap="YlGnBu").set(title='cumPnl')
        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            plt.show()


    @staticmethod
    def profile_rt(pnls, pdf=None, fig_size=(11.69, 8.27)):
        col_list = ['cumReturnGmv', 'cumReturnGmvLong', 'cumReturnGmvShort']
        rt_plot = pnls[col_list].copy()

        lines = rt_plot.plot.line(figsize=fig_size)
        lines.yaxis.set_major_formatter(mat_ticker.FuncFormatter('{0:.0%}'.format))

        plt.title('Profile: Return')
        plt.xticks(rotation=45)
        plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)

        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

        return lines

    def report(
            self,
            backtesting_data: BacktestingDataModel,
            pdf_file_path='',
            pdf_name='',
    ):
        if pdf_name == '':
            pdf_name = f'report_{self._pm}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pdf'

        fig_size = (11.69, 8.27)
        signal_check_res = {}
        with PdfPages(os.path.join(pdf_file_path, pdf_name)) as pdf:
            first_page = plt.figure(figsize=fig_size)
            first_page.clf()
            txt = f'Backtesting: {self._pm}'
            first_page.text(0.5, 0.5, txt, transform=first_page.transFigure, size=24, ha="center")
            pdf.savefig()
            plt.close()

            positions = backtesting_data.positions
            market_data = backtesting_data.marketData
            pnls = backtesting_data.pnls

            # pm profile check
            # drawdown profile
            self.profile_drawdown(pnls, pdf=pdf, fig_size=fig_size)

            self.profile_concentration(positions, pdf=pdf, fig_size=fig_size)

            self.profile_volatility(positions, market_data, pdf=pdf, fig_size=fig_size)

            self.profile_rt(pnls=pnls, pdf=pdf, fig_size=fig_size)

            # best signal back testing
            for signal_col in ['cumReturnGmv', 'cumReturnGmvLong', 'cumReturnGmvShort']:
                for ls in [Direction.LONG, Direction.SHORT]:
                    print(ls)
                    tt = f'{signal_col}_{ls}'
                    signal_res = self.signal_check(
                        backtesting_data=backtesting_data,
                        direction=ls,
                        signal_col=signal_col
                    )
                    signal_check_res[tt] = signal_res
                    title = f'Signal: {signal_col}, Hedge: {ls}'

                    if signal_res is not None and len(signal_res.dailyPnl) > 0:
                        self.perf_metrix_heatmap_plot(
                            daily_pnl=signal_res.dailyPnl,
                            col1='days_to_hedge',
                            col2='new_high_signal_days',
                            pdf=pdf,
                            grid_title=title,
                        )

                    # plt.figure(figsize=fig_size)
                    # sns.heatmap(signal_res.heatData, cmap="YlGnBu").set(title=title)
                    # pdf.savefig()
                    # plt.close()


        return signal_check_res


class DrawdownSodStatusCheck:
    def __init__(
            self,
            stt_dt,
            ed_dt,
    ):
        self.stt_dt = stt_dt
        self.ed_dt = ed_dt

        # get pm list
        self.risk_api = RiskApiDataLoader(token=get_token('minfengy'))
        df_books = self.risk_api.get_books()

        self.pm_list = df_books['name'].drop_duplicates().dropna()


    def new_high_check(
            self,
            local_new_high_range: list,
    ):
        for book in self.pm_list:
            pnls_ls = self.risk_api.get_pnls(
                start_date=self.stt_dt,
                end_date=self.ed_dt,
                group_by='longshort',
                book=book,
            )
            pnls_ls = _process_time_series(hist_pnl=pnls_ls)

            # find new high days
            result_all = []
            pm_list_filtered = []
            for col in ['cumReturnGmv', 'cumReturnBp', 'cumPnl']:
                # ltd drawdown
                pnls_ls[f'{col}HighLtd'] = pnls_ls[col].cummax()
                y = pnls_ls[f'{col}HighLtd'] == pnls_ls[col]
                pnls_ls[f'{col}NewHighDays'] = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

                for d in local_new_high_range:

                    pnls_ls[f'{col}High{d}D'] = pnls_ls[col].rolling(d).max()
                    y = pnls_ls[f'{col}High{d}D'] == pnls_ls[col]
                    pnls_ls[f'{col}High{d}DNewHighDays'] = y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)

                    # pnls['maxDrawdownGmv'] = pnls['cumReturnGmv'].add(1).div(pnls['highReturnGmv'].add(1)) - 1
                    # pnls['maxDrawdownPnl'] = pnls['cumPnl'] - pnls['highPnl']


                result = []

                if col in pnls_ls:
                    new_high = dict()
                    new_high['PM'] = self._pm

                    # gmv return check
                    last_value = list(pnls_ls[col])[-1]
                    for d in local_new_high_range:
                        if last_value == max(list(pnls_ls[col][-d:])):
                            if len(pnls_ls[col]) < d + 1:
                                first_value = 0
                            else:
                                first_value = list(pnls_ls[col])[-d - 1]
                            if col != 'cumPnl':
                                new_high[f'{d}D'] = (last_value + 1) / (first_value + 1) - 1
                            else:
                                new_high[f'{d}D'] = last_value - first_value
                        else:
                            new_high[f'{d}D'] = np.nan

                    if last_value == max(list(pnls_ls[col])):
                        if col != 'cumPnl':
                            new_high['LTD'] = (last_value + 1) / (list(pnls_ls[col])[0] + 1) - 1
                        else:
                            new_high['LTD'] = last_value - list(pnls_ls[col])[0]
                    else:
                        new_high['LTD'] = np.nan

                    result.append(new_high)


def pm_new_high_check():
    """ check pm new high by long short"""
    risk_api = RiskApiDataLoader(token=get_token('minfengy'))
    pnls = risk_api.get_pnls(
        start_date=start_date,
        end_date=end_date,
        group_by='longshort',
        book=book,
        include_aum=True,
        include_bp=True
    )

    pass


def heatmap_plot_one(pm, signal_check_res: SignalCheckResponse):
    # plot heat map
    sns.heatmap(signal_check_res.heatData, cmap="YlGnBu").set(title=pm)

    plt.show()


def heatmap_plot(
    signal_check_res: Dict[str, SignalCheckResponse],
    row,
):
    # plot heat map
    col = int(np.ceil(len(signal_check_res.keys()) / row))
    fig, axes = plt.subplots(col, row, figsize=(18, 28))
    plt.subplots_adjust(wspace=1, hspace=1)
    i = 0

    for nm in list(signal_check_res.keys()):
        sns.heatmap(signal_check_res[nm].heatData, cmap="YlGnBu", ax=axes[i // row, i % row]).set(title=nm)
        i += 1

    plt.show()


def box_plot(cons):
    # show boxplot
    cons_total = pd.concat(cons)
    cons_total_draw = cons_total[~cons_total['pm'].isin(['DSING', 'ECMIPO'])]
    cons_total_draw = cons_total_draw[(cons_total_draw['top5GmvPctLong'] > 0) & (cons_total_draw['top5GmvPctLong'] < 0.9)]
    sns.boxplot(x='top5GmvPctLong', y='pm', data=cons_total_draw).set_title('Top 5 Long Concentration')

    plt.show()


def pdf_gen():
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig1 = plt.figure()
    plt.plot([2, 1, 7, 1, 2], color='red', lw=5)

    fig2 = plt.figure()
    plt.plot([3, 5, 1, 5, 3], color='green', lw=5)

    def save_multi_image(filename):
        pp = PdfPages(filename)
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()


if __name__ == '__main__':
    start_date_mkt = '2019-09-01'
    start_date_pos = '2019-01-01'
    end_date_pos = '2022-08-05'

    # risk_api = RiskApiDataLoader(token=get_token('minfengy'))
    # books = risk_api.get_books()
    #
    # pm_list = books['name'].drop_duplicates().dropna()
    #
    # pm_list = [i for i in pm_list if i not in ['AW_ALPHA', 'xTRADE', 'xHEDGE']]

    signal_check_res = {}
    data = {}
    pm_list = ['EPAUL']
    for pm_name in pm_list:
        print(pm_name)
        f = open('o.pickle', 'rb')
        obj = pickle.load(f)
        for key, val in obj.items():
            obj[key] = val.replace({'--': np.nan})
        data[pm_name] = BacktestingDataModel.parse_obj(obj)
        f.close()
        # try:
        engine = DrawdownHedge(
            pm=pm_name,
            pos_start_date=start_date_pos,
            pos_end_date=end_date_pos,
            mkt_start_date=start_date_mkt,
            mkt_end_date=end_date_pos,
            intraday_mode=False,
        )

        # data[pm_name] = engine.get_data()

        print('Running')
        signal_check_res[pm_name] = engine.report(data[pm_name])

    # heatmap_plot_one('AZHAN', signal_check_res['AZHAN'])

    print('po')
        # except Exception as e:
        #     print(e)
    # heatmap_plot(signal_check_res)







