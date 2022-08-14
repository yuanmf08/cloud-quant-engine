import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.kde import gaussian_kde
from numpy import linspace
from statsmodels.distributions.empirical_distribution import ECDF

from src.settings import NOISE_LEVEL


def cal_batting_slugging(start_date_data, end_date_data):
    col_diff = ['book', 'inceptionPnlUsd', 'underlyingInstrumentId']
    pnl_diff = end_date_data[col_diff].groupby(['book', 'underlyingInstrumentId']).sum().reset_index()

    pnl_diff['pnlPeriod'] = pnl_diff['inceptionPnlUsd']
    if start_date_data is not None and len(start_date_data) > 0:
        merge_data = start_date_data[col_diff].rename(columns={'inceptionPnlUsd': 'inceptionPnlUsdPre'}).groupby(['book', 'underlyingInstrumentId']).sum().reset_index()
        pnl_diff = pnl_diff.merge(merge_data, how='left', left_on=['book', 'underlyingInstrumentId'], right_on=['book', 'underlyingInstrumentId'], suffixes=['', '_y'])
        pnl_diff['pnlPeriod'] = pnl_diff['pnlPeriod'].fillna(0) - pnl_diff['inceptionPnlUsdPre'].fillna(0)

    pnl_diff = pnl_diff[pnl_diff['pnlPeriod'].abs() > NOISE_LEVEL].copy()
    pnl_diff['group'] = np.where(pnl_diff['pnlPeriod'] > 0, 'Winner', 'Loser')
    pnl_diff['count'] = pnl_diff.groupby(['book', 'group'])['underlyingInstrumentId'].transform('count')
    pnl_diff['avg'] = pnl_diff.groupby(['book', 'group'])['pnlPeriod'].transform('mean')

    pnl_diff_pm = pd.pivot_table(pnl_diff, values=['count', 'avg'], index=['book'], columns=['group'], aggfunc='first').reset_index()
    pnl_diff_pm.columns = pnl_diff_pm.columns.map(''.join)

    # if no loser or no winner add columns
    if 'countLoser' not in list(pnl_diff_pm.columns):
        pnl_diff_pm['countLoser'] = 0
        pnl_diff_pm['avgLoser'] = np.NaN

    if 'countWinner' not in list(pnl_diff_pm.columns):
        pnl_diff_pm['countWinner'] = 0
        pnl_diff_pm['avgWinner'] = np.NaN

    pnl_diff_pm['countTotal'] = pnl_diff_pm['countLoser'] + pnl_diff_pm['countWinner']
    pnl_diff_pm['batting'] = pnl_diff_pm['countWinner'].div(pnl_diff_pm['countTotal'])
    pnl_diff_pm['slugging'] = pnl_diff_pm['avgWinner'].div(pnl_diff_pm['avgLoser']).abs()

    # merge batting slugging to data last
    pnl_diff_pm.index = pnl_diff_pm['book']
    return pnl_diff_pm[['batting', 'slugging', 'avgWinner', 'avgLoser']]


def cal_return_mean_sd(df, min_obs, annualization_factor):
    annualized_mean = (1 + df).prod() ** (annualization_factor / len(df)) - 1
    annualized_sd = df.std() * (annualization_factor ** 0.5)
    test = np.NaN
    p_value = np.NaN

    if len(df.dropna()) >= min_obs:
        t_test = stats.ttest_1samp(df, 0, nan_policy='omit')
        test = t_test[0]
        p_value = t_test[1]
    return [annualized_mean, annualized_sd, test, p_value]


def cal_skewness(df, min_obs):
    skewness = df.skew()
    test = np.NaN
    p_value = np.NaN
    if len(df.dropna()) >= max(8, min_obs):
        skew_test = stats.skewtest(df, nan_policy='omit')
        test = skew_test[0]
        p_value = skew_test[1]
    return [skewness, test, p_value]


def cal_kurtosis(df, min_obs):
    kurt = df.kurtosis()
    test = np.NaN
    p_value = np.NaN
    if len(df.dropna()) >= max(8, min_obs):
        kurt_test = stats.kurtosistest(df, nan_policy='omit')
        test = kurt_test[0]
        p_value = kurt_test[1]
    return [kurt, test, p_value]


def cal_factor_attr(barra_rt, factor_list):
    """ calculate factor attribution using barra linking methodology """

    # check if any return < -1
    if (barra_rt['Portfolio'] < -1).sum() > 0:
        return None

    # get cum attribution from barra
    port_return = (1 + barra_rt['Portfolio']).prod() - 1
    benchmark_return = (1 + barra_rt['Benchmark']).prod() - 1

    if benchmark_return == port_return:
        a_log = 1 + benchmark_return
    else:
        a_log = (port_return - benchmark_return) / (math.log(1 + port_return) - math.log(1 + benchmark_return))

    def get_b_log(row):
        port_rt = row['Portfolio']
        bench_rt = row['Benchmark']
        if port_rt == bench_rt:
            return 1 / (1 + bench_rt) * a_log
        else:
            return (math.log(1 + port_rt) - math.log(1 + bench_rt)) / (port_rt - bench_rt) * a_log

    barra_rt['b_log'] = barra_rt.apply(get_b_log, axis=1)

    # get benchmark attribution
    bench_attr = benchmark_return / port_return

    res = defaultdict(dict)

    for factor in factor_list:
        if factor == 'Benchmark':
            res[factor]['return'] = benchmark_return
            res[factor]['perfAttribution'] = bench_attr
        else:
            factor_return_adj = (barra_rt[factor] * barra_rt['b_log']).sum()
            res[factor]['perfAttribution'] = factor_return_adj / port_return
            res[factor]['return'] = (1 + barra_rt[factor]).prod() - 1

    return res


def cal_significant_test(obj, df, min_obs, annualization_factor):
    [obj['annualizedMean'], obj['annualizedSd'], obj['meanTTest'], obj['meanType1Error']] = cal_return_mean_sd(df, min_obs, annualization_factor)
    [obj['skewness'], obj['skewnessTest'], obj['skewnessType1Error']] = cal_skewness(df, min_obs)
    [obj['kurtosis'], obj['kurtosisTest'], obj['kurtosisType1Error']] = cal_kurtosis(df, min_obs)
    return obj


def weekday_count(start_date, end_date, date_format='%Y-%m-%d'):
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)

    start_weekday = start.weekday()
    end_weekday = end.weekday()

    first = start + timedelta(days=6 - start_weekday)  ## end of first week
    last = end - timedelta(days=end_weekday)  ## start of last week
    days = ((last - first).days - 1) * 5 / 7  ## this will always multiply of 7

    days = days + max(5 - start_weekday, 0) + max(end_weekday + 1, 5)

    return days


def cal_annualized_factor(start_date, end_date):
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    workdays = weekday_count(start_date[:4] + '-01-01', end_date[:4] + '-12-31', date_format='%Y-%m-%d')
    return workdays / (end_year - start_year + 1)


def split_month(start_date, end_date, date_format='%Y-%m-%d'):
    # start_date and end_date are datetime
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} is not before end date {end_date}")

    year = start_date.year
    month = start_date.month

    result = []

    while (year, month) <= (end_date.year, end_date.month):
        start_of_month = datetime(year, month, 1)
        end_of_month = (datetime(year, month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        result.append([max(start_date, start_of_month).strftime(date_format), min(end_date, end_of_month).strftime(date_format)])

        if month == 12:
            month = 1
            year += 1
        else:
            month += 1

    return result


def create_e_dist(samples):
    # this create the kernel, given an array it will estimate the probability over that values
    kde = gaussian_kde(samples)
    # these are the values over which your kernel will be evaluated
    dist_space = linspace(min(samples), max(samples), 100)
    # plot the results
    pdf = kde(dist_space)
    cdf = ECDF(samples)(dist_space)
    return [dist_space, pdf, cdf]
