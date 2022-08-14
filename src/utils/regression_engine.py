import warnings
from collections import defaultdict
from typing import List, Optional
from typing import Union

import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
from pandas import DataFrame
from pydantic import BaseModel
from scipy import optimize
from scipy.stats import ttest_ind_from_stats
from scipy.stats.distributions import t
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error

MIN_OBS = 2


class TESTS:
    T_TEST = 'T-Test'
    JARQUE_BERA_TEST = 'Jarque–Bera test'
    F_TEST = 'F-Test'


class LassoCriterion:
    AIC = 'AIC'
    BIC = 'BIC'
    CV = 'CV'
    USER_DEFINED = 'UDL'


class RegModelType:
    OLS = 'OLS'
    LASSO = 'LASSO'
    CURVE_FIT = 'CURVE FIT'


class OlsParamModel(BaseModel):
    name: str
    coef: float
    std: Optional[float] = None
    tValue: Optional[float] = None
    pValue: Optional[float] = None


class JbTestModel(BaseModel):
    jarqueBera: Optional[float] = None
    chiSqTwoTailProb: Optional[float] = None
    skew: Optional[float] = None
    kurtosis: Optional[float] = None


class TTestModel(BaseModel):
    tValue: Optional[float] = None
    pValue: Optional[float] = None


class FitResultModel(BaseModel):
    regModelType: Optional[str] = None
    yName: Optional[str] = None
    params: Optional[List[OlsParamModel]] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    obs: Optional[int] = None
    r2: Optional[float] = None
    r2Adj: Optional[float] = None
    jbTest: Optional[JbTestModel] = None
    residuals: Optional[List[float]] = None
    residualMean: Optional[float] = None
    residualStd: Optional[float] = None
    lassoLambda: Optional[float] = None
    lassoMse: Optional[float] = None
    lassoLarsICMethod: Optional[str] = None
    lassoLarsICAlphas: Optional[List] = None
    lassoLarsICCriterion: Optional[List] = None
    lassoLarsICMses: Optional[List] = None
    lassoLarsICMsesMean: Optional[List] = None


def replace_nan(val):
    if np.isnan(val) or np.isinf(val) or np.isneginf(val):
        return None
    else:
        return val


def func_piecewise_two_step(
        x: float,
        threshold: float,
        y1: float,
        k1: float,
        k2: float
):
    # given the kink condition
    y2 = y1 + k1 * threshold - k2 * threshold
    return np.piecewise(
        x,
        [x <= threshold],
        [
            lambda var: y1 + k1 * var,
            lambda var: y2 + k2 * var
        ]
    )


class RegressionEngine:
    def __init__(
            self,
            data_set: DataFrame = None
    ):
        self._data_set = data_set
        self._INTERCEPT: str = 'Intercept'
        self._jarque_bera_test = None
        if len(self._data_set) < MIN_OBS:
            raise ValueError('No enough obs')
        if 'date' not in list(self._data_set.columns):
            raise ValueError('No date in data set')

        self._include_intercept = True

        self._lasso_cv = 5
        self._lasso_max_iter = 1000
        self._lasso_tolerance = 0.0001
        self._lasso_positive = False

    def set_intercept_name(
            self,
            intercept_name: str = 'Intercept'
    ):
        self._INTERCEPT = intercept_name

    def set_reg_params(
            self,
            include_intercept: bool = True
    ):
        self._include_intercept = include_intercept

    def set_lasso_params(
            self,
            cv: int = 5,
            max_iter: int = 1000,
            tolerance: float = 0.0001,
            positive: bool = False,
    ):
        self._lasso_cv = cv
        self._lasso_max_iter = max_iter
        self._lasso_tolerance = tolerance
        self._lasso_positive = positive

    @staticmethod
    def test_jarque_bera(residuals) -> JbTestModel:
        """
        test for normal distribution
        """

        test = sms.jarque_bera(residuals)

        return JbTestModel(
            jarqueBera=test[0],
            chiSqTwoTailProb=test[1],
            skew=test[2],
            kurtosis=test[3]
        )

    @staticmethod
    def test_t(
            reg_res,
            hypotheses: Union[dict, str],
    ):
        """
        :param reg_res: reg result from statsmodels
        :param hypotheses: key is param name, value is test value
        :return: test result
        """
        if isinstance(hypotheses, str):
            hypotheses_str = hypotheses
            return reg_res.t_test(hypotheses_str)
        else:
            hypotheses_str = ', '.join([f'{key} = {val}' for key, val in hypotheses.items()])

            t_test = reg_res.t_test(hypotheses_str)

            res = defaultdict(dict)
            keys = list(hypotheses.keys())
            for key, val in hypotheses.items():
                index = keys.index(key)
                res[key] = {
                    'tValue': t_test.tvalue[index],
                    'pValue': t_test.pvalue[index]
                }

            return res

    @staticmethod
    def test_t_by_std(
            estimation: float,
            theo_value: float,
            std: float,
            dof: float,
    ) -> TTestModel:
        # student-t value for the dof and confidence level
        if not std or std == 0 or std in [np.nan, np.inf, -np.inf]:
            return TTestModel()

        t_val = (estimation - theo_value) / std

        p_val = t.sf(np.abs(t_val), dof) * 2

        return TTestModel(
            tValue=replace_nan(t_val),
            pValue=replace_nan(p_val),
        )

    @staticmethod
    def test_t_ind_from_stats(
            mean1: float,
            std1: float,
            nobs1: float,
            mean2: float,
            std2: float,
            nobs2: float,
            equal_var: bool = True,
    ) -> TTestModel:
        if not replace_nan(std1) or not replace_nan(std2):
            return TTestModel()

        t_stat = ttest_ind_from_stats(
            mean1=mean1,
            std1=std1,
            nobs1=nobs1,
            mean2=mean2,
            std2=std2,
            nobs2=nobs2,
            equal_var=equal_var,
        )
        return TTestModel(
            tValue=t_stat[0],
            pValue=t_stat[1],
        )

    def reg_ols_multi_factor(
            self,
            y_col: str,
            x_col: Union[str, list],
    ) -> FitResultModel:
        if isinstance(x_col, str):
            x_col = [x_col]

        for col in [y_col] + x_col:
            if col not in self._data_set:
                raise ValueError(f'No {col} data found')

        x = self._data_set[x_col]
        y = self._data_set[y_col]

        if self._include_intercept:
            x = sm.add_constant(x)

        model = sm.OLS(y, x)
        reg_res = model.fit()

        params = list()

        for key, val in reg_res.params.items():
            params.append(OlsParamModel(
                name=self._INTERCEPT if key == 'const' else key,
                coef=replace_nan(val),
                std=None,
                tValue=replace_nan(reg_res.tvalues[key]),
                pValue=replace_nan(reg_res.pvalues[key]),
            ))
        return FitResultModel(
            regModelType=RegModelType.OLS,
            params=params,
            obs=len(y),
            r2=replace_nan(reg_res.rsquared),
            r2Adj=replace_nan(reg_res.rsquared_adj),
            residuals=list(reg_res.resid),
            jbTest=self.test_jarque_bera(reg_res.resid),
            residualMean=reg_res.resid.mean(),
            residualStd=reg_res.resid.std(),
            yName=y_col,
            startDate=self._data_set['date'].min(),
            endDate=self._data_set['date'].max(),
        )

    def reg_piecewise_two_steps(
            self,
            y_col: str,
            x_col: str,
            p0: list,
    ) -> FitResultModel:
        # drop na
        try:
            popt, pcov = optimize.curve_fit(
                func_piecewise_two_step,
                list(self._data_set[x_col]),
                list(self._data_set[y_col]),
                p0=p0,
                maxfev=5000
            )
        except RuntimeError:
            return FitResultModel()

        threshold, y1, k1, k2 = popt
        y2 = y1 + k1 * threshold - k2 * threshold

        # one standard deviation errors on the parameters
        threshold_std, y1_std, k1_std, k2_std = np.sqrt(np.diag(pcov))

        # residuals
        y_data = np.array(self._data_set[y_col])
        x_data = np.array(self._data_set[x_col])
        estimations = np.array([func_piecewise_two_step(i, *list(popt)) for i in list(x_data)])
        residuals = y_data - estimations

        # t test for hypothesis 0
        dof = max(0, len(residuals) - len(popt))
        size = max(0, len(residuals))

        k1_t_test = self.test_t_by_std(k1, 0, k1_std, dof)
        k2_t_test = self.test_t_by_std(k2, 0, k2_std, dof)
        y1_t_test = self.test_t_by_std(y1, 0, y1_std, dof)
        threshold_t_test = self.test_t_by_std(threshold, 0, threshold_std, dof)
        k1_k2_t_test = self.test_t_ind_from_stats(
            mean1=k1,
            std1=k1_std,
            nobs1=size,
            mean2=k2,
            std2=k2_std,
            nobs2=size,
        )

        params = [
            OlsParamModel(
                name=f'{x_col}_1',
                coef=replace_nan(k1),
                std=replace_nan(k1_std),
                tValue=k1_t_test.tValue,
                pValue=k1_t_test.pValue,
            ),
            OlsParamModel(
                name=f'{x_col}_2',
                coef=replace_nan(k2),
                std=replace_nan(k2_std),
                tValue=k2_t_test.tValue,
                pValue=k2_t_test.pValue,
            ),
            OlsParamModel(
                name=f'{self._INTERCEPT}_1',
                coef=replace_nan(y1),
                std=replace_nan(y1_std),
                tValue=y1_t_test.tValue,
                pValue=y1_t_test.pValue,
            ),
            OlsParamModel(
                name=f'{self._INTERCEPT}_2',
                coef=replace_nan(y2),
            ),
            OlsParamModel(
                name='Threshold',
                coef=replace_nan(threshold),
                std=replace_nan(threshold_std),
                tValue=threshold_t_test.tValue,
                pValue=threshold_t_test.pValue,
            ),
            OlsParamModel(
                name=f'{x_col}_1-{x_col}_2',
                coef=replace_nan(k1 - k2),
                tValue=k1_k2_t_test.tValue,
                pValue=k1_k2_t_test.pValue,
            ),
        ]

        # r2
        residual_sum_sqr = np.sum(residuals ** 2)
        total_sum_sqr = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (residual_sum_sqr / total_sum_sqr)
        r_squared_adj = 1 - ((1 - r_squared) * (size - 1) / (dof - 1))

        return FitResultModel(
            yName=y_col,
            regModelType=RegModelType.CURVE_FIT,
            params=params,
            obs=len(self._data_set),
            r2=replace_nan(r_squared),
            r2Adj=replace_nan(r_squared_adj),
            residuals=residuals.tolist(),
            residualMean=residuals.mean(),
            residualStd=residuals.std(),
            startDate=self._data_set['date'].min(),
            endDate=self._data_set['date'].max(),
        )

    def _split_x_y(
            self,
            y_col: str,
            x_col: Union[str, list],
    ):
        if isinstance(x_col, str):
            x_col = [x_col]

        for col in [y_col] + x_col:
            if col not in self._data_set:
                raise ValueError(f'No {col} data found')

        x = self._data_set[x_col]
        y = self._data_set[y_col]

        return x, y

    def reg_lasso_find_best_lambda(
            self,
            y_col: str,
            x_col: Union[str, list],
            criterion: str,  # AIC/BIC/CV
            alphas: np.ndarray = None,
    ):
        x, y = self._split_x_y(x_col=x_col, y_col=y_col)

        model = None
        if criterion in [LassoCriterion.AIC, LassoCriterion.BIC]:
            model = linear_model.LassoLarsIC(
                criterion=criterion.lower(),
                max_iter=self._lasso_max_iter,
                positive=self._lasso_positive,
                normalize=False,
                fit_intercept=self._include_intercept,
            )

        elif criterion == LassoCriterion.CV and self._lasso_cv <= len(self._data_set):
            model = linear_model.LassoCV(
                alphas=alphas,
                cv=self._lasso_cv,
                fit_intercept=self._include_intercept,
                tol=self._lasso_tolerance,
                max_iter=self._lasso_max_iter,
                positive=self._lasso_positive,
            )

        if model:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=ConvergenceWarning)
                try:
                    model.fit(x, y)
                except ConvergenceWarning as e:
                    # object not converge
                    warnings.filterwarnings('ignore', category=ConvergenceWarning)
                    model.fit(x, y)

            model.fit(x, y)
            return model

    def reg_lasso_lars_ic(
            self,
            y_col: str,
            x_col: Union[str, list],
            criterion: str,
    ) -> FitResultModel:
        """
        find optimal lambda based on ic method AIC/BIC/CV/UDL
        and run lasso regression on optimal lambda

        :param y_col:
        :param x_col:
        :param criterion: AIC/BIC
        :return:
        """
        x, y = self._split_x_y(x_col=x_col, y_col=y_col)
        if criterion not in [LassoCriterion.AIC, LassoCriterion.BIC]:
            raise ValueError('No IC Criterion specified')

        # find best alpha:
        model_alpha = self.reg_lasso_find_best_lambda(
            y_col=y_col,
            x_col=x_col,
            criterion=criterion,
        )

        if model_alpha:
            # use the best alpha to do lasso reg
            res = self.reg_lasso_with_lambda(
                y_col=y_col,
                x_col=x_col,
                alpha=model_alpha.alpha_,
            )

            # add additional info
            res.lassoLarsICMethod = criterion
            res.lassoLarsICAlphas = model_alpha.alphas_.tolist()
            res.lassoLarsICCriterion = model_alpha.criterion_.tolist()

            return res

    def reg_lasso_cv(
            self,
            y_col: str,
            x_col: Union[str, list],
            alphas: np.ndarray = None,
    ) -> Union[FitResultModel, float]:
        """
        find optimal lambda based on ic method AIC/BIC/CV/UDL
        and run lasso regression on optimal lambda

        :param alphas:
        :param y_col:
        :param x_col:
        :return:
        """
        x, y = self._split_x_y(x_col=x_col, y_col=y_col)
        if self._lasso_cv <= len(self._data_set):
            # find best alpha:
            model_alpha = self.reg_lasso_find_best_lambda(
                y_col=y_col,
                x_col=x_col,
                criterion=LassoCriterion.CV,
                alphas=alphas,
            )

            if model_alpha:
                res = self.reg_lasso_with_lambda(
                    y_col=y_col,
                    x_col=x_col,
                    alpha=model_alpha.alpha_,
                )

                # add additional info
                res.lassoLarsICMethod = LassoCriterion.CV
                res.lassoLarsICAlphas = model_alpha.alphas_.tolist()
                res.lassoLarsICMses = model_alpha.mse_path_.tolist()
                res.lassoLarsICMsesMean = model_alpha.mse_path_.mean(axis=-1).tolist()

                return res

    def find_lasso_alpha_with_num_params(
            self,
            y_col: str,
            x_col: Union[str, list],
            params_size: int = None,
    ) -> float:
        x, y = self._split_x_y(x_col=x_col, y_col=y_col)

        # find optimal lambda given num of parameters
        al_ceil = 1
        al_floor = 0
        al = (al_ceil + al_floor) / 2
        while 0 <= al <= 1:
            lasso_model = linear_model.Lasso(
                alpha=al,
                max_iter=self._lasso_max_iter,
                tol=self._lasso_tolerance,
                positive=self._lasso_positive,
                fit_intercept=self._include_intercept,
            )
            lasso_model.fit(x, y)
            n_zero = np.count_nonzero(lasso_model.coef_)

            if n_zero < params_size:
                al_ceil = al
                al = (al + al_floor) / 2
            elif n_zero > params_size:
                al_floor = al
                al = (al + al_ceil / 2)
            else:
                return al

    def reg_lasso_ols_with_num_params(
            self,
            y_col: str,
            x_col: Union[str, list],
            params_size: int = None,
    ) -> FitResultModel:
        """
        Use lasso to find the best N params and use ols to do estimation and test
        :param y_col:
        :param x_col:
        :param params_size:
        :return:
        """
        x, y = self._split_x_y(x_col=x_col, y_col=y_col)
        alpha = self.find_lasso_alpha_with_num_params(
            x_col=x_col,
            y_col=y_col,
            params_size=params_size
        )

        if alpha:
            lasso_model = linear_model.Lasso(
                alpha=alpha,
                max_iter=self._lasso_max_iter,
                tol=self._lasso_tolerance,
                positive=self._lasso_positive,
                fit_intercept=self._include_intercept,
            )
            lasso_model.fit(x, y)

            arr = np.nonzero(lasso_model.coef_)
            x_filtered = x.iloc[:, arr[0]]

            res_ols = self.reg_ols_multi_factor(
                y_col=y.name,
                x_col=list(x_filtered.columns),
            )

            return res_ols

    def reg_lasso_with_num_params(
            self,
            y_col: str,
            x_col: Union[str, list],
            params_size: int = None,
    ) -> FitResultModel:
        """
        Use lasso to find the best N params and use ols to do estimation and test
        :param y_col:
        :param x_col:
        :param params_size:
        :return:
        """
        x, y = self._split_x_y(x_col=x_col, y_col=y_col)
        alpha = self.find_lasso_alpha_with_num_params(
            x_col=x_col,
            y_col=y_col,
            params_size=params_size
        )

        if alpha:
            res_ols = self.reg_lasso_with_lambda(
                y_col=y_col,
                x_col=x_col,
                alpha=alpha,
            )

            return res_ols


    def reg_lasso_ols_with_alpha(
            self,
            y_col: str,
            x_col: Union[str, list],
            alpha: float = None,
    ) -> FitResultModel:
        """
        Use lasso to find the best N params and use ols to do estimation and test
        :param alpha:
        :param y_col:
        :param x_col:
        :return:
        """
        # find best alpha using lasso
        res = self.reg_lasso_with_lambda(
            y_col=y_col,
            x_col=x_col,
            alpha=alpha,
        )

        non_zero = [i.name for i in res.params if i.name != 'Intercept' and i.coef != 0]
        res_ols = FitResultModel(regModelType=RegModelType.OLS)
        if len(non_zero) > 0:
            res_ols = self.reg_ols_multi_factor(
                y_col=y_col,
                x_col=non_zero,
            )

        return res_ols

    def reg_lasso_with_lambda(
            self,
            y_col: str,
            x_col: Union[str, list],
            alpha: float = None,
    ) -> FitResultModel:
        x, y = self._split_x_y(x_col=x_col, y_col=y_col)

        lasso = linear_model.Lasso(
            alpha=alpha,
            max_iter=self._lasso_max_iter,
            tol=self._lasso_tolerance,
            positive=self._lasso_positive,
            fit_intercept=self._include_intercept,
        )
        lasso.fit(x, y)

        score = lasso.score(x, y)
        y_pred = lasso.predict(x)
        mse = mean_squared_error(y, y_pred)

        coefs = list()

        for key, val in dict(zip(x_col, lasso.coef_)).items():
            coefs.append(OlsParamModel(
                name=key,
                coef=val
            ))

        if self._include_intercept:
            coefs.append(OlsParamModel(
                name='Intercept',
                coef=lasso.intercept_
            ))

        return FitResultModel(
            regModelType=RegModelType.LASSO,
            yName=y_col,
            obs=len(y),
            params=coefs,
            r2=score,
            lassoLambda=lasso.alpha,
            lassoMse=mse,
            startDate=self._data_set['date'].min(),
            endDate=self._data_set['date'].max(),
        )
