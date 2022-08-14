import pandas as pd
import random
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from pandas import DataFrame
from jax.scipy.stats import poisson
from jax.tree_util import tree_flatten
from scipy.stats.kde import gaussian_kde
from numpy import linspace
import itertools
from src.models.return_based_analysis.simulator_model import (
    ModelPriorGuess,
    StochasticModel,
    McmcSettings,
    RegressionMethod,
    McmcResult,
    RegressionModel,
    SimulationRunSettings,
    SimParamMethod,
    SimJobResponse,
    SampleDist,
    SimPageSettings,
    McmcSummaryModel,
)
from statsmodels.distributions.empirical_distribution import ECDF
from src.utils.stochastic_calc.stochastic_models import (
    DcMsvModelHyperParamGuess,
    AFactorMsvModelHyperParamGuess,
    MFactorMsvModelHyperParamGuess,
    BasicModelHyperParamGuess,
    JumpDiffModelHyperParamGuess,
    HestonModelHyperParamGuess,
    BayesianLinearRegHyperParamGuess,
    OrnsteinUhlenbeckModelHyperParamGuess,
    BayesianPiecewiseHyperParamGuess,
)
from src.utils.stochastic_calc.prob_models import (
    IntParams,
    numpyro_sample,
    DimensionType
)


MIN_OBS = 2

RAND_SEED_SIZE = 10000000

numpyro.set_host_device_count(4)


def flatten(l):
    for i in l:
        if isinstance(i, list) or isinstance(i, np.ndarray) or isinstance(i, jnp.ndarray):
            yield from flatten(list(i))
        else:
            yield i


class StochasticProcess:
    def __init__(
            self,
            data_set: DataFrame = pd.DataFrame(),
            device: str = 'cpu',
    ):
        numpyro.set_platform(device)
        self._data_set = data_set

        self._annual_factor = 262
        self._reg_obs_prefix = 'obsReg'
        self._sto_obs_prefix = 'obsSto'

        self._inference_model = {

            StochasticModel.basic: self.mcmc_basic,
            StochasticModel.heston: self.mcmc_heston,
            # StochasticModel.hullWhite: self.mcmc_heston_mcmc_sde,
            StochasticModel.ornsteinUhlenbeck: self.mcmc_ornstein_uhlenbeck,
            StochasticModel.mertonJump: self.mcmc_jump_diffusion,
            StochasticModel.dcMsv: self.mcmc_dc_msv,
            StochasticModel.aFactorMsv: self.mcmc_a_factor_msv,
            StochasticModel.mFactorMsv: self.mcmc_m_factor_msv,

            RegressionModel.capm: self.mcmc_linear_reg,
            RegressionModel.multiFactor: self.mcmc_linear_reg_multi,
            RegressionModel.piecewiseCapm: self.mcmc_linear_reg_piecewise,
        }

    def _data_check(self):
        if len(self._data_set) < MIN_OBS:
            raise ValueError('No enough obs')

    def inference_mcmc(
            self,
            model: StochasticModel,
            guess: ModelPriorGuess,
            mcmc_settings: McmcSettings,
            sim_page_settings: SimPageSettings,
            model_param_only: bool = True
    ) -> McmcResult:
        self._data_check()

        nuts_kernel = NUTS(
            self._inference_model[model],
            init_strategy=init_to_median,
        )

        mcmc = MCMC(
            nuts_kernel,
            num_warmup=mcmc_settings.warmupSteps,
            num_samples=mcmc_settings.numSamples,
            num_chains=mcmc_settings.numChains,
        )

        # MCMC can only run in main process
        mcmc.run(
            jrd.PRNGKey(random.randint(0, RAND_SEED_SIZE)),
            guess,
        )

        # get sample and extra info
        samples = mcmc.get_samples(group_by_chain=True)
        extra = mcmc.get_extra_fields(group_by_chain=True)

        if not isinstance(samples, dict):
            samples = {
                "Param:{}".format(i): v for i, v in enumerate(tree_flatten(samples)[0])
            }

        # get all sites result
        sites = mcmc._states[mcmc._sample_field]
        sites_filtered = sites

        # keep only params in the guess
        if model_param_only:
            display_name_list = [val.displayName for val in list(guess.hyperParamGuess.values())]
            param_list = display_name_list + [i for i in list(samples.keys()) if i.startswith('extra')]
            samples = {k: v for k, v in samples.items() if k in param_list}
            sites_filtered = {k: v for k, v in sites.items() if k in param_list}

        # create diagnostics summary
        summary = numpyro.diagnostics.summary(sites_filtered, group_by_chain=True)

        # create sample distribution
        def create_sample_dist(sample_obj, summary_obj):
            dist_res = dict()
            summary_res = []
            for key, val in sample_obj.items():
                if key in guess.hyperParamGuess.keys():
                    dist_by_factor = {}
                    model_param_type = guess.hyperParamGuess[key]
                    if model_param_type.dimensionType == DimensionType.One:
                        data_dist = self._create_e_dist(val)
                        dist_by_factor[key] = SampleDist(
                            name=key,
                            x=data_dist[0],
                            pdf=data_dist[1],
                            cdf=data_dist[2],
                        )

                        # add summary
                        summary_res.append(
                            McmcSummaryModel(
                                key=key,
                                subKey=key,
                                summary=summary_obj[key]
                            )
                        )
                    elif model_param_type.dimensionType == DimensionType.Params:
                        factor_num = range(len(sim_page_settings.factorSettings.factors))
                        flat_lst = list(itertools.chain(*val))
                        min_num = min(flat_lst)
                        max_num = max(flat_lst)
                        for i in factor_num:
                            factor_name = sim_page_settings.factorSettings.factors[i].name
                            data_dist = self._create_e_dist(val[i], min_num=min_num, max_num=max_num)
                            dist_by_factor[factor_name] = SampleDist(
                                name=f'{key}_{factor_name}',
                                x=data_dist[0],
                                pdf=data_dist[1],
                                cdf=data_dist[2],
                            )
                            summary_res.append(
                                McmcSummaryModel(
                                    key=key,
                                    subKey=factor_name,
                                    summary={k: v[i] for k, v in summary_obj[key].items()}
                                )
                            )
                    elif model_param_type.dimensionType == DimensionType.Corr:
                        factor_num = range(len(sim_page_settings.factorSettings.factors))

                        flat_lst = list(itertools.chain(*list(itertools.chain(*val))))
                        min_num = min(flat_lst)
                        max_num = max(flat_lst)
                        for i in factor_num:
                            for j in factor_num:
                                if j > i:
                                    corr_name = f'Corr: {sim_page_settings.factorSettings.factors[i].name}_{sim_page_settings.factorSettings.factors[j].name}'
                                    corr_samples = [x[i][j] for x in val]
                                    data_dist = self._create_e_dist(corr_samples, min_num=min_num, max_num=max_num)
                                    dist_by_factor[corr_name] = SampleDist(
                                        name=f'{key}_{corr_name}',
                                        x=data_dist[0],
                                        pdf=data_dist[1],
                                        cdf=data_dist[2],
                                    )
                                    summary_res.append(
                                        McmcSummaryModel(
                                            key=key,
                                            subKey=corr_name,
                                            summary={k: v[i][j] for k, v in summary_obj[key].items()}
                                        )
                                    )

                    elif model_param_type.dimensionType == DimensionType.HyperFactor:
                        hyper_factor_num = range(val[0])

                        flat_lst = list(itertools.chain(*val))
                        min_num = min(flat_lst)
                        max_num = max(flat_lst)
                        for i in hyper_factor_num:
                            hyper_factor_name = f'Factor {i}'
                            data_dist = self._create_e_dist([x[i] for x in val], min_num=min_num, max_num=max_num)
                            dist_by_factor[hyper_factor_name] = SampleDist(
                                name=f'{key}_{hyper_factor_name}',
                                x=data_dist[0],
                                pdf=data_dist[1],
                                cdf=data_dist[2],
                            )
                            summary_res.append(
                                McmcSummaryModel(
                                    key=key,
                                    subKey=hyper_factor_name,
                                    summary={k: v[i] for k, v in summary_obj[key].items()}
                                )
                            )
                    dist_res[key] = dist_by_factor
            return dist_res, summary_res

        # summary for chains merged
        sample_merged = {k: list(itertools.chain(*v)) for k, v in samples.items()}
        sample_dist, summary_filtered = create_sample_dist(sample_merged, summary)
        # build summary group by chain:
        summary_filtered_by_chain = []
        sample_dist_by_chain = []
        for c in range(mcmc_settings.numChains):
            sites_grouped = {k: val[c] for k, val in sites_filtered.items()}
            summary_by_chain = numpyro.diagnostics.summary(sites_grouped, group_by_chain=False)
            sample_grouped = {k: val[c] for k, val in samples.items()}
            sample_dist_single, summary_filtered_single = create_sample_dist(sample_grouped, summary_by_chain)
            sample_dist_by_chain.append(sample_dist_single)
            summary_filtered_by_chain.append(summary_filtered_single)

        # convert jnp array to list for aws upload
        for key, val in samples.items():
            samples[key] = val.tolist()

        for key, val in extra.items():
            extra[key] = val.tolist()

        mcmc_res = McmcResult(
            samples=samples,
            summary=summary_filtered,
            summaryByChain=summary_filtered_by_chain,
            extra=extra,
            sampleDist=sample_dist,
            sampleDistByChain=sample_dist_by_chain,
        )

        # mcmc.print_summary(exclude_deterministic=False)

        return mcmc_res

    def mcmc_simulation(
            self,
            sto_calib_res: SimJobResponse,
            asset_rt_calib_res: SimJobResponse,
            sim_run_settings: SimulationRunSettings,
    ) -> dict:
        def create_mcmc_posterior_samples(num_sim, raw_samples: dict, sim_param_method: SimParamMethod, sim_param_override: dict):
            # create samples for simulation
            post_samples = {}
            item_one = list(raw_samples.values())[0]
            shuffle_samples = 0
            if sim_param_method == SimParamMethod.allPost:
                shuffle_samples = jrd.choice(jrd.PRNGKey(random.randint(0, RAND_SEED_SIZE)), jnp.array(jnp.arange(len(item_one))), shape=(num_sim,))
            for k, v in raw_samples.items():
                if sim_param_method == SimParamMethod.allPost:
                    post_samples[k] = jnp.array(v)[shuffle_samples]
                elif sim_param_method == SimParamMethod.mean:
                    post_samples[k] = jnp.repeat(jnp.array([jnp.mean(v)]), num_sim, axis=0)
                elif sim_param_method == SimParamMethod.median:
                    post_samples[k] = jnp.repeat(jnp.array([jnp.median(v)]), num_sim, axis=0)
                elif sim_param_method == SimParamMethod.override:
                    post_samples[k] = jnp.repeat(jnp.array([sim_param_override[k]]), num_sim, axis=0)

            return post_samples

        sim_res = {}

        num_params = len(asset_rt_calib_res.simPageSettings.factorSettings.factors)

        # model assembling
        sto_model = self._inference_model[sto_calib_res.stoCalibSettings.modelPriorGuess.model]
        asset_rt_model = self._inference_model[asset_rt_calib_res.assetReturnCalibSettings.modelPriorGuess.model]

        # this is a combined model (stochastic + regression) for prediction
        def model_assembled(
                sto_guess: ModelPriorGuess,
                asset_rt_guess: ModelPriorGuess,
        ):
            independent_vars = sto_model(
                guess=sto_guess,
                sim_run_settings=sim_run_settings,
                run_sim=True,
                num_params=num_params
            )
            return asset_rt_model(
                guess=asset_rt_guess,
                sim_run_settings=sim_run_settings,
                run_sim=True,
                independent_vars=independent_vars,
            )

        # get sto mcmc result
        sto_mcmc_res = list(sto_calib_res.stoCalibResult.values())[0]
        # chains to use for simulation
        sto_chain_list = jnp.array(sim_run_settings.stoCalibChainList)

        # get samples and flatten
        sto_samples = sto_mcmc_res.samples
        for key, val in sto_samples.items():
            sto_samples[key] = jnp.concatenate(jnp.array(val)[sto_chain_list])

        # random pick posterior samples
        sto_posterior_samples = create_mcmc_posterior_samples(
            sim_run_settings.numSim,
            sto_samples,
            sim_run_settings.stoSimParamMethod,
            sim_run_settings.stoSimParamOverride,
        )

        # run prediction for sto calib
        sto_predictive = numpyro.infer.Predictive(sto_model, posterior_samples=sto_posterior_samples)
        sto_pred_res = sto_predictive(
            jrd.PRNGKey(random.randint(0, RAND_SEED_SIZE)),
            sto_calib_res.stoCalibSettings.modelPriorGuess,
            sim_run_settings,
            True,
            num_params,
        )

        # store factor simulation results
        factor_list = [i.name for i in sto_calib_res.simPageSettings.factorSettings.factors]
        sto_res_lst = sto_pred_res[self._sto_obs_prefix].tolist()
        for i in range(len(factor_list)):
            f = factor_list[i]
            f_res = []
            for s in range(sim_run_settings.numSim):
                sim_s = sto_res_lst[s]
                if len(factor_list) == 1:
                    f_res.append([None if jnp.isnan(m) else m for m in sim_s])
                else:
                    f_res.append([None if jnp.isnan(m[i]) else m[i] for m in sim_s])
            sim_res[f'Factor {f}'] = f_res

        # asset rt resampling
        asset_rt_res = asset_rt_calib_res.assetReturnCalibResult
        for ast_rt_key, ast_rt_val in asset_rt_res.items():
            asset_rt_samples = ast_rt_val.samples

            # flatten samples
            if asset_rt_calib_res.assetReturnCalibSettings.inferenceMethod == RegressionMethod.MCMC:
                asset_rt_chain_list = jnp.array(sim_run_settings.assetReturnCalibChainList)
                for key, val in asset_rt_samples.items():
                    asset_rt_samples[key] = jnp.concatenate(jnp.array(val)[asset_rt_chain_list])

            asset_posterior_samples = create_mcmc_posterior_samples(
                sim_run_settings.numSim,
                asset_rt_samples,
                sim_run_settings.assetRtSimParamMethod,
                sim_run_settings.assetRtSimParamOverride,
            )

            # combine resample result
            posterior_samples = sto_posterior_samples | sto_pred_res | asset_posterior_samples

            predictive = numpyro.infer.Predictive(model_assembled, posterior_samples=posterior_samples)
            pred = predictive(
                jrd.PRNGKey(random.randint(0, RAND_SEED_SIZE)),
                sto_calib_res.stoCalibSettings.modelPriorGuess,
                asset_rt_calib_res.assetReturnCalibSettings.modelPriorGuess
            )

            # nan to none for json uploading
            lst = pred[self._reg_obs_prefix].tolist()
            for index, l in enumerate(lst):
                a = [None if jnp.isnan(i) else i for i in l]
                lst[index] = a

            sim_res[ast_rt_key] = lst

        return sim_res

    def mcmc_linear_reg(
            self,
            guess: ModelPriorGuess,
            sim_run_settings: SimulationRunSettings = None,
            run_sim: bool = False,
            independent_vars: jnp.array = None,
            **kwargs,
    ):
        """ Bayesian linear regression """
        if run_sim:
            num_obs = sim_run_settings.numObs
            dependent_vars = None
        else:
            dependent_col = ['y']
            dependent_vars, num_obs, num_params = self._data_shape(sim_run_settings, run_sim, 1, self._data_set[dependent_col])
            independent_cols = [i for i in list(self._data_set.columns) if i != 'y']
            independent_vars, _n, _m = self._data_shape(sim_run_settings, run_sim, 1, self._data_set[independent_cols])

        hyper_param_guess: BayesianLinearRegHyperParamGuess = BayesianLinearRegHyperParamGuess.parse_obj(guess.hyperParamGuess)
        intercept = numpyro_sample(hyper_param_guess.interceptReg)
        sigma = numpyro_sample(hyper_param_guess.sigmaReg)
        coefs = numpyro_sample(hyper_param_guess.coefsReg)
        measurement_error = numpyro_sample(hyper_param_guess.measurementErrorReg)

        with numpyro.plate('estReg', num_obs):
            estimates = numpyro.sample(
                'estimatesReg',
                dist.Normal(intercept + independent_vars * coefs, sigma)
            )

        with numpyro.plate('dataReg', num_obs):
            obs = numpyro.sample(
                self._reg_obs_prefix,
                dist.Normal(estimates, measurement_error),
                obs=dependent_vars,
            )

        return obs

    def mcmc_linear_reg_multi(
            self,
            guess: ModelPriorGuess,
            sim_run_settings: SimulationRunSettings = None,
            run_sim: bool = False,
            independent_vars: jnp.array = None,
            **kwargs,
    ):
        """ Bayesian linear multi factor regression """
        if run_sim:
            num_obs = sim_run_settings.numObs
            dependent_vars = None
        else:
            # remove nan
            _data_set = self._data_set.dropna()

            num_obs = len(_data_set)
            dependent_col = 'y'
            dependent_vars = jnp.array(_data_set[dependent_col])
            independent_cols = [i for i in list(_data_set.columns) if i != 'y']
            independent_vars = jnp.array(_data_set[independent_cols])
        num_params = len(independent_vars[0])

        hyper_param_guess: BayesianLinearRegHyperParamGuess = BayesianLinearRegHyperParamGuess.parse_obj(guess.hyperParamGuess)

        intercept = numpyro_sample(hyper_param_guess.interceptReg)
        sigma = numpyro_sample(hyper_param_guess.sigmaReg)
        measurement_error = numpyro_sample(hyper_param_guess.measurementErrorReg)

        with numpyro.plate('paramReg', num_params):
            coefs = numpyro_sample(hyper_param_guess.coefsReg)

        linear_part = intercept + jnp.matmul(independent_vars, coefs)

        estimates = numpyro.sample(
            'estimatesReg',
            dist.Normal(linear_part, sigma)
        )

        with numpyro.plate('dataReg', num_obs):
            obs = numpyro.sample(
                self._reg_obs_prefix,
                dist.Normal(
                    estimates,
                    measurement_error
                ),
                obs=dependent_vars,
            )

        return obs

    def mcmc_linear_reg_piecewise(
            self,
            guess: ModelPriorGuess,
            sim_run_settings: SimulationRunSettings = None,
            run_sim: bool = False,
            independent_vars: jnp.array = None,
            **kwargs,
    ):
        """ Bayesian linear regression """
        if run_sim:
            num_obs = sim_run_settings.numObs
            dependent_vars = None
        else:
            dependent_col = ['y']
            dependent_vars, num_obs, num_params = self._data_shape(sim_run_settings, run_sim, 1, self._data_set[dependent_col])
            independent_cols = [i for i in list(self._data_set.columns) if i != 'y']
            independent_vars, _n, _m = self._data_shape(sim_run_settings, run_sim, 1, self._data_set[independent_cols])

        hyper_param_guess: BayesianPiecewiseHyperParamGuess = BayesianPiecewiseHyperParamGuess.parse_obj(guess.hyperParamGuess)

        intercept1 = numpyro_sample(hyper_param_guess.intercept1Reg)
        coef1 = numpyro_sample(hyper_param_guess.coef1Reg)
        coef2 = numpyro_sample(hyper_param_guess.coef2Reg)
        threshold = numpyro_sample(hyper_param_guess.thresholdReg)
        sigma = numpyro_sample(hyper_param_guess.sigmaReg)
        measurement_error = numpyro_sample(hyper_param_guess.measurementErrorReg)

        # given the kink condition
        intercept2 = numpyro.deterministic(hyper_param_guess.intercept2Reg.displayName, intercept1 + coef1 * threshold - coef2 * threshold)

        switch_0 = jnp.where(independent_vars <= threshold, 0, 1)
        switch_1 = jnp.where(independent_vars > threshold, 1, 0)

        linear_part = (intercept1 + independent_vars * coef1) * switch_0 + (intercept2 + independent_vars * coef2) * switch_1

        estimates = numpyro.sample(
            'estimatesReg',
            dist.Normal(linear_part, sigma)
        )

        with numpyro.plate('dataReg', num_obs):
            obs = numpyro.sample(
                self._reg_obs_prefix,
                dist.Normal(
                    estimates,
                    measurement_error
                ),
                obs=dependent_vars,
            )

        return obs

    def mcmc_basic(
            self,
            guess: ModelPriorGuess,
            sim_run_settings: SimulationRunSettings = None,
            run_sim: bool = False,
            num_params: int = None,
            **kwargs,
    ):
        data_obs, num_obs, num_params = self._data_shape(sim_run_settings, run_sim, num_params, self._data_set)

        # define param samples
        hyper_param_guess: BasicModelHyperParamGuess = BasicModelHyperParamGuess.parse_obj(guess.hyperParamGuess)
        drift = numpyro_sample(hyper_param_guess.drift)
        drift_daily = jnp.true_divide(drift, self._annual_factor)
        sigma = numpyro_sample(hyper_param_guess.sigma)
        sigma_daily = jnp.true_divide(sigma, jnp.sqrt(self._annual_factor))
        measurement_error = numpyro_sample(hyper_param_guess.measurementError)

        with numpyro.plate('data', num_obs):
            wiener_comp = numpyro.sample(
                'wiener',
                dist.Normal(
                    0,
                    sigma_daily
                ),
            )

            obs = numpyro.sample(
                self._sto_obs_prefix,
                dist.Normal(
                    drift_daily + wiener_comp,
                    measurement_error
                ),
                obs=data_obs,
            )

        return obs

    def mcmc_jump_diffusion(
            self,
            guess: ModelPriorGuess,
            sim_run_settings: SimulationRunSettings = None,
            run_sim: bool = False,
            num_params: int = None,
            **kwargs,
    ):
        data_obs, num_obs, num_params = self._data_shape(sim_run_settings, run_sim, num_params, self._data_set, log_rt=True)

        # define param samples
        hyper_param_guess: JumpDiffModelHyperParamGuess = JumpDiffModelHyperParamGuess.parse_obj(guess.hyperParamGuess)

        drift = numpyro_sample(hyper_param_guess.drift)
        jump_lam = numpyro_sample(hyper_param_guess.jumpLam)
        jump_lam_daily = jnp.true_divide(jump_lam, self._annual_factor)
        jump_size_mean = numpyro_sample(hyper_param_guess.jumpSizeMean)
        sigma = numpyro_sample(hyper_param_guess.sigma)
        sigma_daily = jnp.true_divide(sigma, jnp.sqrt(self._annual_factor))

        jump_size_std = numpyro_sample(hyper_param_guess.jumpSizeStd)
        measurement_error = numpyro_sample(hyper_param_guess.measurementError)

        max_extreme_value = 15
        p_list = jnp.zeros(max_extreme_value)
        for i in range(max_extreme_value):
            p_list = p_list.at[i].set(poisson.pmf(k=i, mu=jump_lam_daily))

        with numpyro.plate('data', num_obs):
            wiener_comp = numpyro.sample(
                'wiener',
                dist.Normal(
                    0,
                    sigma_daily
                ),
            )

            num_jumps = numpyro.sample(
                f'num_jumps',
                dist.Categorical(p_list),
                infer={'enumerate': 'parallel'},
            )

            jumped_size = numpyro.sample(
                f'jumped_size',
                dist.Normal(
                    jump_size_mean,
                    jump_size_std,
                ),
            )

            jump_comp = jnp.multiply(num_jumps, jumped_size)

            drift_comp = jnp.true_divide((drift - sigma ** 2 / 2 - jump_lam * (jump_size_mean + jump_size_std ** 2 * 0.5)), self._annual_factor)

            log_obs = numpyro.sample(
                self._sto_obs_prefix,
                dist.Normal(drift_comp + wiener_comp + jump_comp, measurement_error),
                obs=data_obs,
            )

        obs = jnp.exp(log_obs) - 1

        return obs

    def mcmc_heston(
            self,
            guess: ModelPriorGuess,
            sim_run_settings: SimulationRunSettings = None,
            run_sim: bool = False,
            num_params: int = None,
            **kwargs,
    ):
        data_obs, num_obs, num_params = self._data_shape(sim_run_settings, run_sim, num_params, self._data_set)

        dt = 1 / self._annual_factor

        # define param samples
        hyper_param_guess: HestonModelHyperParamGuess = HestonModelHyperParamGuess.parse_obj(guess.hyperParamGuess)

        drift = numpyro_sample(hyper_param_guess.drift)
        long_var = numpyro_sample(hyper_param_guess.longVar)
        var_zero = numpyro_sample(hyper_param_guess.varZero)
        kappa = numpyro_sample(hyper_param_guess.kappa)
        vega = numpyro_sample(hyper_param_guess.vega)
        measurement_error = numpyro_sample(hyper_param_guess.measurementError)
        rho = numpyro_sample(hyper_param_guess.rho)

        wiener_cov = jnp.array([[1, rho], [rho, 1]])

        def trans_fn(v_t, t):
            wiener_proc = numpyro.sample(
                'wiener',
                dist.MultivariateNormal(
                    loc=jnp.zeros(2),
                    covariance_matrix=wiener_cov
                )
            ) * jnp.sqrt(dt)
            wiener_stock = wiener_proc[0]
            wiener_vol = wiener_proc[1]

            rt = drift* dt + jnp.sqrt(v_t) * wiener_stock
            v_t_1 = jnp.maximum(v_t + kappa * (long_var - v_t) * dt + vega * jnp.sqrt(v_t) * wiener_vol, 0.00001)

            return v_t_1, rt

        _, rt_res = scan(trans_fn, var_zero, xs=None, length=num_obs)

        with numpyro.plate('data', num_obs):
            obs = numpyro.sample(
                self._sto_obs_prefix,
                dist.Normal(rt_res, measurement_error),
                obs=data_obs
            )

        return obs

    def mcmc_ornstein_uhlenbeck(
            self,
            guess: ModelPriorGuess,
            sim_run_settings: SimulationRunSettings = None,
            run_sim: bool = False,
            num_params: int = None,
            **kwargs,
    ):
        data_obs, num_obs, num_params = self._data_shape(sim_run_settings, run_sim, num_params, self._data_set)

        dt = 1 / self._annual_factor

        # define param samples
        hyper_param_guess: OrnsteinUhlenbeckModelHyperParamGuess = OrnsteinUhlenbeckModelHyperParamGuess.parse_obj(guess.hyperParamGuess)

        sample_log_var_mean = numpyro_sample(hyper_param_guess.logVarMean)

        # phi is daily reversion factor
        sample_phi = numpyro_sample(hyper_param_guess.phi)
        sample_sigma_eta = numpyro_sample(hyper_param_guess.sigmaEta)
        measurement_error = numpyro_sample(hyper_param_guess.measurementError)

        def trans_fn(log_var, t):
            sample_eta = numpyro.sample(
                'eta_matrix',
                dist.Normal(
                    0,
                    sample_sigma_eta * dt,
                )
            )

            log_var_t = sample_log_var_mean + jnp.multiply(sample_phi, log_var - sample_log_var_mean) + sample_eta

            daily_rt = numpyro.sample(
                'daily_rt',
                dist.Normal(
                    0,
                    log_var_t,
                ),
            )

            return log_var_t, daily_rt

        _, daily_rt_series = scan(trans_fn, jnp.zeros(num_params), xs=None, length=num_obs)

        with numpyro.plate('data', num_obs, dim=-2):
            obs = numpyro.sample(
                self._sto_obs_prefix,
                dist.Normal(
                    daily_rt_series,
                    measurement_error,
                ),
                obs=data_obs,
            )

        return obs

    def mcmc_dc_msv(
            self,
            guess: ModelPriorGuess,
            sim_run_settings: SimulationRunSettings = None,
            run_sim: bool = False,
            num_params: int = None,
            **kwargs,
    ):
        """
        Multivariate model
        https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=1359&context=soe_research
        Dynamic correlation-MSV (Model 5)

        Note: Guess is annualized, obs is daily, convert annualized to daily before cal
        """

        data_obs, num_obs, num_params = self._data_shape(sim_run_settings, run_sim, num_params, self._data_set, log_rt=True)

        # define param samples
        hyper_param_guess: DcMsvModelHyperParamGuess = DcMsvModelHyperParamGuess.parse_obj(guess.hyperParamGuess)

        with numpyro.plate('params', num_params):
            # std of return
            sigma_mean = numpyro_sample(hyper_param_guess.sigmaMean)

            # initial std of return
            sigma_zero = numpyro_sample(hyper_param_guess.sigmaZero)

            # log variance of return for mean reversion time variation
            log_var_mean = jnp.log(jnp.power(sigma_mean, 2))

            # reversion factor for log variance
            phi = numpyro_sample(hyper_param_guess.phi)

            # residual error of log variance
            sigma_eta = numpyro_sample(hyper_param_guess.sigmaEta)

            # residual error for model setting
            measurement_error = numpyro_sample(hyper_param_guess.measurementError)

        # unconditional correlation
        corr_mean = numpyro.sample(
            hyper_param_guess.corrMean.displayName,
            dist.LKJ(
                num_params,
                1,
            )
        )

        corr_zero = numpyro.sample(
            'corrZero',
            dist.LKJ(
                num_params,
                1,
            )
        )

        # alpha factor for correlation mean reversion
        corr_alpha = numpyro_sample(hyper_param_guess.corrAlpha)

        # beta factor for correlation mean reversion
        corr_beta = numpyro_sample(hyper_param_guess.corrBeta)

        # create daily return time series
        def daily_rt_series_scan(carry, t):
            (log_var, q, e) = carry

            eta = numpyro.sample(
                'eta',
                dist.Normal(
                    jnp.zeros(num_params),
                    sigma_eta,
                )
            )

            # mean reversion part for log var
            log_var_t = log_var_mean + jnp.multiply(phi, log_var - log_var_mean) + eta

            # mean reversion part for q factor and correlation
            q_t = corr_mean + jnp.multiply(corr_alpha, jnp.outer(e, e) - corr_mean) + jnp.multiply(corr_beta, q - corr_mean)
            q_t_diag = jnp.diag(q_t)
            corr = jnp.true_divide(q_t, jnp.sqrt(jnp.outer(q_t_diag, q_t_diag)))

            # daily covariance matrix
            std_t_daily = jnp.sqrt(jnp.true_divide(jnp.exp(log_var_t), self._annual_factor))
            cov = jnp.multiply(corr, jnp.outer(std_t_daily, std_t_daily))

            daily_rt = numpyro.sample(
                f'daily_rt',
                dist.MultivariateNormal(
                    loc=0,
                    covariance_matrix=cov,
                ),
            )

            e_t = jnp.multiply(daily_rt, jnp.power(std_t_daily, -1))

            return (log_var_t, q_t, e_t), daily_rt

        _, daily_rt_series = scan(daily_rt_series_scan, (sigma_zero, corr_zero, jnp.zeros(num_params)), xs=None, length=num_obs)

        with numpyro.plate('data', num_obs, dim=-2):
            with numpyro.plate('paramData', num_params):
                obs = numpyro.sample(
                    self._sto_obs_prefix,
                    dist.Normal(
                        daily_rt_series,
                        measurement_error,
                    ),
                    obs=data_obs,
                )
        normal_rt = jnp.exp(obs) - 1
        return normal_rt

    def mcmc_a_factor_msv(
            self,
            guess: ModelPriorGuess,
            sim_run_settings: SimulationRunSettings = None,
            run_sim: bool = False,
            num_params: int = None,
            **kwargs,
    ):
        data_obs, num_obs, num_params = self._data_shape(sim_run_settings, run_sim, num_params, self._data_set, log_rt=True)

        # define param samples
        hyper_param_guess: AFactorMsvModelHyperParamGuess = AFactorMsvModelHyperParamGuess.parse_obj(guess.hyperParamGuess)
        factor_param: IntParams = hyper_param_guess.factorNum.distParam
        factor_num = factor_param.value

        with numpyro.plate('hyperFactors', factor_num):
            factor_sigma_mean = numpyro_sample(hyper_param_guess.factorSigmaMean)

            # log variance of return for mean reversion time variation
            factor_log_var_mean = jnp.log(jnp.power(factor_sigma_mean, 2))

            # phi is daily reversion factor
            factor_phi = numpyro_sample(hyper_param_guess.factorPhi)

            factor_sigma_eta = numpyro_sample(hyper_param_guess.factorSigmaEta)

            with numpyro.plate('fL', num_params):
                factor_loading = numpyro_sample(hyper_param_guess.factorLoading)

        with numpyro.plate('params', num_params):
            sigma_epsilon = numpyro_sample(hyper_param_guess.sigmaEpsilon)
            measurement_error = numpyro_sample(hyper_param_guess.measurementError)

        # create daily return time series
        def daily_rt_series_scan(carry, t):
            factor_log_var = carry

            mu = numpyro.sample(
                'mu',
                dist.Normal(
                    jnp.zeros(factor_num),
                    1,
                )
            )

            eta = numpyro.sample(
                'eta',
                dist.Normal(
                    jnp.zeros(factor_num),
                    factor_sigma_eta,
                )
            )

            factor_log_var_t = factor_log_var_mean + jnp.multiply(factor_phi, factor_log_var - factor_log_var_mean) + eta
            f_t = jnp.sqrt(jnp.true_divide(jnp.exp(factor_log_var_t), self._annual_factor)) * mu

            daily_rt = numpyro.sample(
                'daily_rt',
                dist.Normal(
                    jnp.matmul(factor_loading, f_t),
                    sigma_epsilon
                )
            )

            return factor_log_var_t, daily_rt

        _, daily_rt_series = scan(daily_rt_series_scan, jnp.zeros(factor_num), xs=None, length=num_obs)

        with numpyro.plate('data', num_obs, dim=-2):
            with numpyro.plate('paramData', num_params):
                obs = numpyro.sample(
                    self._sto_obs_prefix,
                    dist.Normal(
                        daily_rt_series,
                        measurement_error,
                    ),
                    obs=data_obs,
                )
        normal_rt = jnp.exp(obs) - 1
        return normal_rt

    def mcmc_m_factor_msv(
            self,
            guess: ModelPriorGuess,
            sim_run_settings: SimulationRunSettings = None,
            run_sim: bool = False,
            num_params: int = None,
            **kwargs,
    ):
        data_obs, num_obs, num_params = self._data_shape(
            sim_run_settings,
            run_sim,
            num_params,
            self._data_set,
            log_rt=True
        )

        # define param samples
        hyper_param_guess: MFactorMsvModelHyperParamGuess = MFactorMsvModelHyperParamGuess.parse_obj(guess.hyperParamGuess)

        sigma_mean = numpyro_sample(hyper_param_guess.sigmaMean)
        log_var_mean = jnp.log(jnp.power(sigma_mean, 2))

        phi = numpyro_sample(hyper_param_guess.phi)
        corr = numpyro_sample(hyper_param_guess.corr, num_params=num_params)

        sigma_eta = numpyro_sample(hyper_param_guess.sigmaEta)

        with numpyro.plate('param', num_params):
            sigma_multiply = numpyro_sample(hyper_param_guess.sigmaMultiply)
            measurement_error = numpyro_sample(hyper_param_guess.measurementError)

        sigma_multiply = jnp.true_divide(sigma_multiply, sigma_multiply[0])

        # create daily return time series
        def daily_rt_series_scan(carry, t):
            log_var = carry

            eta = numpyro.sample('eta', dist.Normal(0, sigma_eta))

            # mean reversion part for log var
            log_var_t = log_var_mean + jnp.multiply(phi, log_var - log_var_mean) + eta

            # daily covariance matrix
            daily_sigma_t = jnp.multiply(
                jnp.sqrt(jnp.true_divide(jnp.exp(log_var_t), self._annual_factor)),
                sigma_multiply
            )
            cov = jnp.multiply(corr, jnp.outer(daily_sigma_t, daily_sigma_t))

            daily_rt = numpyro.sample('daily_rt', dist.MultivariateNormal(loc=0, covariance_matrix=cov))

            return log_var_t, daily_rt

        _, daily_rt_series = scan(daily_rt_series_scan, log_var_mean, xs=None, length=num_obs)

        with numpyro.plate('data', num_obs, dim=-2):
            with numpyro.plate('paramData', num_params):
                obs = numpyro.sample(
                    self._sto_obs_prefix,
                    dist.Normal(
                        daily_rt_series,
                        measurement_error,
                    ),
                    obs=data_obs,
                )

        normal_rt = jnp.exp(obs) - 1

        return normal_rt

    @staticmethod
    def _data_shape(
            sim_run_settings: SimulationRunSettings,
            run_sim: bool,
            num_params: int,
            data,
            log_rt=False,
    ):
        if run_sim:
            num_obs = sim_run_settings.numObs
            data_obs = None
        else:
            if len(data.columns) == 1:
                data_obs = jnp.array(data[list(data.columns)[0]].values)
                (num_obs, num_params) = (len(data), 1)
            else:
                data_obs = jnp.array(data.values.tolist())
                (num_obs, num_params) = data_obs.shape
            if log_rt:
                data_obs = jnp.log(data_obs + 1)

        return data_obs, num_obs, num_params

    @staticmethod
    def _create_e_dist(samples, min_num=None, max_num=None):
        if not min_num:
            min_num = min(samples)

        if not max_num:
            max_num = max(samples)

        # this create the kernel, given an array it will estimate the probability over that values
        kde = gaussian_kde(samples)
        # these are the values over which your kernel will be evaluated
        dist_space = linspace(min_num, max_num, 100)
        # plot the results
        pdf = kde(dist_space)
        cdf = ECDF(samples)(dist_space)
        return [list(dist_space), list(pdf), list(cdf)]
