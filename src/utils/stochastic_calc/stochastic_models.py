from pydantic import BaseModel

from src.utils.stochastic_calc.prob_models import ModelParamType


class DcMsvModelHyperParamGuess(BaseModel):
    sigmaMean: ModelParamType  # mu
    sigmaZero: ModelParamType  # mu
    phi: ModelParamType  # is the rate at which νt reverts to θ
    corrAlpha: ModelParamType
    corrBeta: ModelParamType
    corrMean: ModelParamType
    sigmaEta: ModelParamType  # var for random walk for stock var
    measurementError: ModelParamType


class AFactorMsvModelHyperParamGuess(BaseModel):
    sigmaEpsilon: ModelParamType  # mu
    factorSigmaMean: ModelParamType  # mu
    factorNum: ModelParamType
    factorLoading: ModelParamType
    factorPhi: ModelParamType  # is the rate at which νt reverts to θ
    factorSigmaEta: ModelParamType  # var for random walk for stock var
    measurementError: ModelParamType


class MFactorMsvModelHyperParamGuess(BaseModel):
    sigmaMean: ModelParamType  # mu
    phi: ModelParamType  # is the rate at which νt reverts to θ
    sigmaEta: ModelParamType  # var for random walk for stock var
    corr: ModelParamType  # var for random walk for stock var
    sigmaMultiply: ModelParamType  # var for random walk for stock var
    measurementError: ModelParamType


class BayesianLinearRegHyperParamGuess(BaseModel):
    interceptReg: ModelParamType
    coefsReg: ModelParamType
    sigmaReg: ModelParamType
    measurementErrorReg: ModelParamType


class BayesianPiecewiseHyperParamGuess(BaseModel):
    intercept1Reg: ModelParamType
    intercept2Reg: ModelParamType
    coef1Reg: ModelParamType
    coef2Reg: ModelParamType
    thresholdReg: ModelParamType
    sigmaReg: ModelParamType
    measurementErrorReg: ModelParamType


class BasicModelHyperParamGuess(BaseModel):
    drift: ModelParamType
    sigma: ModelParamType  # annual standard deviation , for weiner process
    measurementError: ModelParamType


class JumpDiffModelHyperParamGuess(BaseModel):
    drift: ModelParamType
    jumpLam: ModelParamType  # intensity of jump i.e. number of jumps per annum
    jumpSizeMean: ModelParamType  # mean of jump size
    jumpSizeStd: ModelParamType  # std of jump size
    sigma: ModelParamType  # annual standard deviation , for weiner process
    measurementError: ModelParamType  # annual standard deviation , for weiner process


class HestonModelHyperParamGuess(BaseModel):
    drift: ModelParamType
    longVar: ModelParamType  # long variance
    varZero: ModelParamType  # long variance
    kappa: ModelParamType  # is the rate at which νt reverts to θ
    vega: ModelParamType  # volatility of the instantaneous variance
    rho: ModelParamType  # brownian motion correlation
    measurementError: ModelParamType


class OrnsteinUhlenbeckModelHyperParamGuess(BaseModel):
    logVarMean: ModelParamType  # mu
    logVarZero: ModelParamType
    phi: ModelParamType  # is the rate at which νt reverts to θ
    sigmaEta: ModelParamType  # var for random walk for stock var
    measurementError: ModelParamType

# class HullWhiteParams(BaseModel):
#     drift = 'drift'
#     longVar = 'longVar'  # long variance
#     varZero = 'varZero'  # long variance
#     kappa = 'kappa'  # is the rate at which νt reverts to θ
#     vega = 'vega'  # volatility of the instantaneous variance
#     rho = 'rho'  # brownian motion correlation
#     error = 'error'


# class OrnsteinUhlenbeckParams(BaseModel):
#     drift = 'drift'
#     theta = 'theta'  # long variance
#     vega = 'vega'  # volatility of the instantaneous variance
#     error = 'error'

