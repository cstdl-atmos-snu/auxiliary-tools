import xarray as xr
import numpy as np
import metpy.calc as mpcalc
import metpy.units as mpunits

from tcpyPI import pi as tcpi
from scipy.optimize import minimize
from scipy.special import gammaln

from .. import constants
from ..calc import sphere
from ..calc import vertical


def model_TCGI_CRH(params, x):
    """
    Computes the model:
    ln y = b + b_eta * eta + b_CRH * CRH + b_PI * PI + b_VWS * VWS + ln cos (latitude)
    """
    b, b_eta, b_CRH, b_PI, b_VWS = params
    eta = np.abs(x["eta"] * 1e5).clip(0, 3.7)
    CRH = x["CRH"] * 100
    PI = x["PI"]
    VWS = x["VWS"]
    ln_cos_lat = np.log(np.cos(np.deg2rad(x["latitude"])))
    ln_pred_y = b + b_eta * eta + b_CRH * CRH + b_PI * PI + b_VWS * VWS + ln_cos_lat
    return np.exp(ln_pred_y)


def objective_TCGI_CRH(params, y, x):
    pred_y = model_TCGI_CRH(params, x)
    L = np.sum(y * np.log(pred_y) - pred_y - gammaln(y + 1))
    loss = -L
    return loss

    # # Usage:
    # y = IBTrACS_hist["tcg"].sel(latitude=slice(-60, 60))
    # x = ds[["eta", "CRH", "PI", "VWS"]].sel(latitude=slice(-60, 60))

    # initial_guess = [-10.0, 1.0, 0.1, 0.5, -0.1]

    # result = minimize(
    #     objective_TCGI_CRH,
    #     initial_guess,
    #     args=(y, x),
    #     method="Nelder-Mead",
    #     options={"disp": True},
    # )
    # TCGI_CRH = model_TCGI_CRH(result.x, x)


def model_TCGI_SD(params, x):
    """
    Computes the model:
    ln y = b + b_eta * eta + b_SD * SD + b_PI * PI + b_VWS * VWS + ln cos (latitude)
    """
    b, b_eta, b_SD, b_PI, b_VWS = params
    eta = np.abs(x["eta"] * 1e5).clip(0, 3.7)
    SD = x["SD"]
    PI = x["PI"]
    VWS = x["VWS"]
    ln_cos_lat = np.log(np.cos(np.deg2rad(x["latitude"])))
    ln_pred_y = b + b_eta * eta + b_SD * SD + b_PI * PI + b_VWS * VWS + ln_cos_lat
    return np.exp(ln_pred_y)


def objective_TCGI_SD(params, y, x):
    pred_y = model_TCGI_SD(params, x)
    L = np.sum(y * np.log(pred_y) - pred_y - gammaln(y + 1))
    loss = -L
    return loss


def model_GPI_without_c(x):
    """
    Computes the model:
    GPI = |10^5 * eta|^(3/2) * (r600 / 50)^3 * (PI / 70)^3 * (1 + 0.1 * VWS)^(-2)
    """
    eta = np.abs(x["eta"] * 1e5)
    r600 = x["r"] / 50
    PI = x["PI"] / 70
    VWS = 1 + 0.1 * x["VWS"]
    temp_pred_y = (eta ** (3 / 2)) * (r600**3) * (PI**3) * (VWS ** (-2))
    return temp_pred_y


def model_GPI_calc_c(y, temp_y_pred):
    mean_yearly_temp_y_pred = (
        temp_y_pred.groupby("time.month")
        .mean("time")
        .sum(["month", "latitude", "longitude"])
    )
    mean_yearly_y = (
        y.groupby("time.month").mean("time").sum(["month", "latitude", "longitude"])
    )

    c = mean_yearly_y.values / mean_yearly_temp_y_pred.values
    return c


def model_GPI(param, x):
    eta = np.abs(x["eta"] * 1e5)
    r600 = x["r"] / 50
    PI = x["PI"] / 70
    VWS = 1 + 0.1 * x["VWS"]
    pred_y = param * (eta ** (3 / 2)) * (r600**3) * (PI**3) * (VWS ** (-2))
    return pred_y


def model_GPI_Xi_without_c(x):
    """
    Computes the model:
    GPI = |eta|^3 * (chi)^(-4/3) * max[(PI - 35), 0]^2 * (25 + VWS)^(-4)
    """
    eta = np.abs(x["eta"])
    chi = x["chi"]
    PI = x["PI"] - 35
    PI = PI.clip(0, None)
    VWS = 25 + x["VWS"]
    temp_pred_y = (eta**3) * (chi ** (-4 / 3)) * (PI**2) * (VWS ** (-4))
    return temp_pred_y


def model_GPI_Xi_calc_c(y, temp_y_pred):
    mean_yearly_temp_y_pred = (
        temp_y_pred.groupby("time.month")
        .mean("time")
        .sum(["month", "latitude", "longitude"])
    )
    mean_yearly_y = (
        y.groupby("time.month").mean("time").sum(["month", "latitude", "longitude"])
    )

    c = mean_yearly_y.values / mean_yearly_temp_y_pred.values
    return c


def model_GPI_Xi(param, x):
    eta = np.abs(x["eta"])
    chi = x["chi"]
    PI = x["PI"] - 35
    PI = PI.clip(0, None)
    VWS = 25 + x["VWS"]
    pred_y = param * (eta**3) * (chi ** (-4 / 3)) * (PI**2) * (VWS ** (-4))
    return pred_y


def model_GPIv_without_c(param, x):
    """
    Computes the model y = (c * vPI * eta)^a * cos(lat) * dx * dy
    """
    dlat = (
        x["latitude"].diff("latitude", label="lower")
        + x["latitude"].diff("latitude", label="upper")
    ).sel(latitude=slice(-60, 60)) / 2
    dlon = (
        x["longitude"].diff("longitude", label="lower")
        + x["longitude"].diff("longitude", label="upper")
    ) / 2
    x = x.sel(latitude=slice(-60, 60))

    vPI = x["vPI"]
    eta = np.abs(x["eta"]).clip(None, 3.7e-5)
    temp_y_pred = (vPI * eta) ** param * np.cos(np.deg2rad(x["latitude"])) * dlat * dlon
    return temp_y_pred


def model_GPIv_without_c(param, x):
    """
    Computes the model y = (c * vPI * eta)^a * cos(lat) * dx * dy
    """
    dlat = (
        x["latitude"].diff("latitude", label="lower")
        + x["latitude"].diff("latitude", label="upper")
    ).sel(latitude=slice(-60, 60)) / 2
    dlon = (
        x["longitude"].diff("longitude", label="lower")
        + x["longitude"].diff("longitude", label="upper")
    ) / 2
    x = x.sel(latitude=slice(-60, 60))

    vPI = x["vPI"]
    eta = np.abs(x["eta"]).clip(None, 3.7e-5)
    temp_y_pred = (vPI * eta) ** param * np.cos(np.deg2rad(x["latitude"])) * dlat * dlon
    return temp_y_pred


def model_GPIv_calc_c(param, y, temp_y_pred):
    mean_yearly_temp_y_pred = (
        temp_y_pred.groupby("time.month")
        .mean("time")
        .sum(["month", "latitude", "longitude"])
    )
    mean_yearly_y = (
        y.groupby("time.month").mean("time").sum(["month", "latitude", "longitude"])
    )

    c = (mean_yearly_y.values / mean_yearly_temp_y_pred.values) ** (1 / param)
    return c


def objective_GPIv(param, y, x):
    """
    Computes the sum of squared errors between the model predictions and the data.

    Returns:
        Sum of squared errors.
    """
    temp_y_pred = model_GPIv_without_c(param, x)
    c = model_GPIv_calc_c(param, y, temp_y_pred)
    y_pred = c**param * temp_y_pred

    clim_y = y.groupby("time.month").mean("time")
    clim_y_pred = y_pred.groupby("time.month").mean("time")
    return np.sum((clim_y - clim_y_pred) ** 2).values


def model_GPIv(params, x):
    """
    Computes the model y = (c * vPI * eta)^a * cos(lat) * dx * dy
    """
    dlat = (
        x["latitude"].diff("latitude", label="lower")
        + x["latitude"].diff("latitude", label="upper")
    ).sel(latitude=slice(-60, 60)) / 2
    dlon = (
        x["longitude"].diff("longitude", label="lower")
        + x["longitude"].diff("longitude", label="upper")
    ) / 2
    x = x.sel(latitude=slice(-60, 60))

    a, c = params
    vPI = x["vPI"]
    eta = np.abs(x["eta"]).clip(None, 3.7e-5)
    y_pred = (c * vPI * eta) ** a * np.cos(np.deg2rad(x["latitude"])) * dlat * dlon
    return y_pred
