from enum import Enum
from typing import Union

import numpyro.distributions as dist
import jax.numpy as jnp
from pydantic import BaseModel
import numpyro


class DistType(str, Enum):
    Normal = 'Normal'
    AffineBeta = 'AffineBeta'
    Uniform = 'Uniform'
    MultivariateNormal = 'MultivariateNormal'
    HalfCauchy = 'HalfCauchy'
    Exponential = 'Exponential'
    LogNormal = 'LogNormal'
    Int = 'Int'
    Lkj = 'Lkj'


class NormalParams(BaseModel):
    distName: DistType
    mean: float
    std: float


class AffineBetaParams(BaseModel):
    distName: DistType
    alpha: float
    beta: float
    loc: float
    scale: float


class UniformParams(BaseModel):
    distName: DistType
    low: float
    high: float


class HalfCauchyParams(BaseModel):
    distName: DistType
    scale: float


class ExponentialParams(BaseModel):
    distName: DistType
    rate: float


class LogNormalParams(BaseModel):
    distName: DistType
    mean: float
    std: float


class IntParams(BaseModel):
    distName: DistType
    value: int


class LkjParams(BaseModel):
    distName: DistType
    con: float


DistParamType = Union[NormalParams, AffineBetaParams, UniformParams, HalfCauchyParams, ExponentialParams, LogNormalParams, IntParams, LkjParams]


class DimensionType(str, Enum):
  Params = 'Params',
  One = 'One',
  Corr = 'Corr',
  HyperFactor = 'HyperFactor',


class ModelParamType(BaseModel):
    displayName: str
    dimensionType: DimensionType
    distParam: DistParamType


def numpyro_sample(model_param_type: ModelParamType, num_params: int = 0, name_override=None):
    if name_override:
        sample_name = name_override
    else:
        sample_name = model_param_type.displayName

    return numpyro.sample(
        sample_name,
        parse_dist_numpyro(
            dist_guess=model_param_type.distParam,
            num_params=num_params,
        )
    )


def parse_dist_numpyro(
        dist_guess: DistParamType,
        num_params: int = 0,
):
    if dist_guess.distName == DistType.Normal:
        return dist.Normal(
            dist_guess.mean,
            dist_guess.std,
        )

    elif dist_guess.distName == DistType.AffineBeta:
        return dist.TransformedDistribution(
            dist.Beta(jnp.array(dist_guess.alpha), jnp.array(dist_guess.beta)),
            dist.transforms.AffineTransform(jnp.array(dist_guess.loc), jnp.array(dist_guess.scale))

        )

    elif dist_guess.distName == DistType.Uniform:
        return dist.Uniform(
            dist_guess.low,
            dist_guess.high,
        )

    elif dist_guess.distName == DistType.HalfCauchy:
        return dist.HalfCauchy(
            dist_guess.scale,
        )

    elif dist_guess.distName == DistType.Exponential:
        return dist.Exponential(
            dist_guess.rate,
        )

    elif dist_guess.distName == DistType.LogNormal:
        return dist.LogNormal(
            dist_guess.mean,
            dist_guess.std,
        )

    elif dist_guess.distName == DistType.Lkj:
        return dist.LKJ(
            num_params,
            dist_guess.con,
        )
