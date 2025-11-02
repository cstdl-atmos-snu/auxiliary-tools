import numpy as np
import xarray as xr
import scipy.fft as fft
import sys
import os

# Append the parent directory to the system path
parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(parent_dir)

import constants

Re = constants.Re
g = constants.g


def haversine_distance(point1, point2):
    """
    Calculate the distance between point1 and point2 using the Haversine formula.

    Parameters:
    - point1, point2: Tuples or arrays representing (latitude, longitude).
      There are three cases:
        1) Both point1 and point2 are single points: returns the distance (a scalar).
        2) One of the points is an array of points: returns an array of distances.
        3) Both points are arrays (point1 has m points and point2 has n points):
           returns a distance matrix of size (m, n).

    The function handles the reshaping of points and conversion from degrees to radians.
    """

    point1 = np.squeeze(np.asarray(point1))
    point2 = np.squeeze(np.asarray(point2))

    single_point1 = point1.ndim == 1
    single_point2 = point2.ndim == 1

    if single_point1 and single_point2:
        point1 = point1.reshape(1, -1)
        point2 = point2.reshape(1, -1)
    elif single_point1:
        point1 = np.tile(point1, (len(point2), 1))
    elif single_point2:
        point2 = np.tile(point2, (len(point1), 1))
    else:
        return distance_matrix_haver(point1, point2)

    lat1, lon1 = point1[:, 0], point1[:, 1]
    lat2, lon2 = point2[:, 0], point2[:, 1]

    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a + 1e-10))

    # Calculate the distance
    distance = Re * c

    return distance


def distance_matrix_haver(point1, point2):
    point1 = np.squeeze(np.asarray(point1))
    point2 = np.squeeze(np.asarray(point2))

    single_point1 = point1.ndim == 1
    single_point2 = point2.ndim == 1

    if single_point1 and single_point2:
        point1 = point1.reshape(1, -1)
        point2 = point2.reshape(1, -1)

    dist_matrix = np.zeros((len(point1), len(point2)))
    for i, point1 in enumerate(point1):
        dist_matrix[i, :] = haversine_distance(point1, point2)
    return dist_matrix


def dx_to_dlon(dx, lat):
    return dx * 360 / (2 * np.pi * Re * np.cos(np.deg2rad(lat)))


def dy_to_dlat(dy):
    return dy * 360 / (2 * np.pi * Re)


# Function to calculate a point given latitude, longitude, bearing, and distance
def calculate_point(lat, lon, bearing, distance):
    lat, lon, bearing = np.radians([lat, lon, bearing])
    new_lat = np.arcsin(
        np.sin(lat) * np.cos(distance / Re)
        + np.cos(lat) * np.sin(distance / Re) * np.cos(bearing)
    )
    new_lon = lon + np.arctan2(
        np.sin(bearing) * np.sin(distance / Re) * np.cos(lat),
        np.cos(distance / Re) - np.sin(lat) * np.sin(new_lat),
    )
    return np.degrees(new_lat), np.degrees(new_lon)


# Function to generate concentric circles
def concentric_circles(center_lat, center_lon, distances, bearing):
    circles = {}
    for distance in distances:
        circle_points = []
        for i_b in bearing:
            point = calculate_point(center_lat, center_lon, i_b, distance)
            circle_points.append(point)
        circles[distance] = circle_points
    return circles


# calculate the area in km^2 of a circle given a radius on the Earth in km
# https://en.wikipedia.org/wiki/Spherical_cap
def shperical_cap_area(radius):
    return 2 * np.pi * Re * Re * (1 - np.cos(radius / Re))


def rolling_sum_np(np_array, win_size=(41, 41), doLonPad=False):
    """
    np_array = (lat, lon)
    """
    # rolling mean 10 by 10 degree (41 by 41 grid, including center)
    lat_win_size = win_size[0]
    lon_win_size = win_size[1]

    if doLonPad:
        np_array_copy = np.zeros(
            (np_array.shape[0], np_array.shape[1] + lon_win_size - 1)
        )
        np_array_copy[:, : lon_win_size // 2] = np_array[:, -(lon_win_size // 2) :]
        np_array_copy[:, lon_win_size // 2 : -(lon_win_size // 2)] = np_array
        np_array_copy[:, -(lon_win_size // 2) :] = np_array[:, : lon_win_size // 2]
    else:
        np_array_copy = np_array.copy()

    data_bar = np.zeros(
        (
            lon_win_size * lat_win_size,
            np_array_copy.shape[0] - lat_win_size + 1,
            np_array_copy.shape[1] - lon_win_size + 1,
        )
    )

    for j in range(lat_win_size):
        for k in range(lon_win_size):
            data_bar[j * lon_win_size + k, :, :] = np_array_copy[
                j : np_array_copy.shape[0] - lat_win_size + j + 1,
                k : np_array_copy.shape[1] - lon_win_size + k + 1,
            ]

    data_bar = np.nansum(data_bar, axis=0)
    return data_bar


def column_integral(data_array):
    """
    Compute the vertical column integral of an xarray DataArray using the midpoint method.

    Parameters:
        data_array (xarray.DataArray): The input data array which must have a vertical coordinate
                                       named "level" (in hPa).

    Returns:
        xarray.DataArray: The column-integrated value.
    """

    # Calculate pressure differences between levels in Pa (1 hPa = 100 Pa)
    dp = data_array["level"].diff(dim="level") * 100

    # Compute the difference of the data values along the vertical dimension
    dvalue = data_array.diff(dim="level")

    # Estimate the midpoint values using the adjacent differences
    mid_value = data_array - 0.5 * dvalue

    # Compute the column integral by summing over the 'level' dimension
    integral = (mid_value * dp).sum(dim="level") / g

    return integral


def dx_central(data_array, doLonPad=False):
    """
    Compute the zonal derivative (with respect to longitude) of an xarray DataArray using a central difference scheme.

    The derivative is calculated along the "longitude" dimension. The grid spacing (dx) is determined from the
    difference between adjacent longitude values (assumed to be uniform) and is adjusted for latitude by
    converting degrees to a physical distance using Earth's mean radius.

    For boundary points:
      - If do_lon_pad is True, periodic boundary conditions are applied using a roll operation.
      - Otherwise, forward and backward differences are used at the boundaries with appropriate scaling.

    Parameters:
        data (xarray.DataArray): The input data array with "longitude" and "latitude" coordinates.
        do_lon_pad (bool): If True, apply periodic boundary conditions in the longitude direction.
                           If False, use forward/backward differences at the boundaries. Defaults to False.

    Returns:
        xarray.DataArray: The computed zonal derivative (df/dx) of the input data.
    """

    # Compute the difference in longitude (assumes uniform spacing)
    dlon = data_array["longitude"][1] - data_array["longitude"][0]
    dx = np.deg2rad(dlon) * np.cos(np.deg2rad(data_array["latitude"])) * Re

    if doLonPad:
        # Apply periodic boundary conditions using roll
        dvalue = data_array.roll(longitude=-1) - data_array.roll(longitude=1)
    else:
        dvalue = data_array.shift(longitude=-1, fill_value=np.nan) - data_array.shift(
            longitude=1, fill_value=np.nan
        )
        dvalue[dict(longitude=0)] = (
            data_array.isel(longitude=1).values - data_array.isel(longitude=0).values
        ) * 2
        dvalue[dict(longitude=-1)] = (
            data_array.isel(longitude=-1).values - data_array.isel(longitude=-2).values
        ) * 2

    # (f(x+dx) - f(x-dx))/(2*dx)
    derivative = dvalue / (2 * dx)

    return derivative


def dy_central(data_array):
    """
    Compute the meridional derivative (with respect to latitude) of an xarray DataArray using a central difference scheme.

    The derivative is calculated along the "latitude" dimension. The grid spacing (dy) is determined from the
    difference between adjacent latitude values (assumed to be uniform) and is adjusted for latitude by
    converting degrees to a physical distance using Earth's mean radius.

    For boundary points:
      - Forward and backward differences are used at the boundaries with appropriate scaling.

    Parameters:
        data (xarray.DataArray): The input data array with "latitude" coordinates.

    Returns:
        xarray.DataArray: The computed zonal derivative (df/dy) of the input data.
    """

    # Compute the difference in longitude (assumes uniform spacing)
    dlat = data_array["latitude"][1] - data_array["latitude"][0]
    dy = np.deg2rad(dlat) * Re

    dvalue = data_array.shift(latitude=-1, fill_value=np.nan) - data_array.shift(
        latitude=1, fill_value=np.nan
    )
    dvalue[dict(latitude=0)] = (
        data_array.isel(latitude=1).values - data_array.isel(latitude=0).values
    ) * 2
    dvalue[dict(latitude=-1)] = (
        data_array.isel(latitude=-1).values - data_array.isel(latitude=-2).values
    ) * 2

    # (f(y+dy) - f(y-dy))/(2*dy)
    derivative = dvalue / (2 * dy)

    return derivative


def spectral_analysis(
    da,
    time_name="time",
    lon_name="lon",
    lat_name="lat",
    window_days=96,
    overlap_days=60,
    hann_days=5,
    num_smooth=15,
):
    """
    Assuming `da` has a regular time, lat, lon grid, and lat dimension is symmetric about the equator.
    """
    dt = da[time_name][1] - da[time_name][0]  # assume regular time axis
    dlon = da[lon_name][1] - da[lon_name][0]  # assume regular lon axis
    samples_per_day = np.timedelta64(1, "D") / dt.values
    samples_per_lon = 360.0 / dlon.values

    nseg = int(window_days * samples_per_day)
    nlon = da.sizes[lon_name]
    nlat = da.sizes[lat_name]

    step = int((window_days - overlap_days) * samples_per_day)  # 36 days
    hann_width = int(hann_days * samples_per_day)

    # symmetric and asymmetric array
    da = da.transpose(time_name, lat_name, lon_name).sortby(lat_name)
    sym_values = (da.values + da.values[:, ::-1, :]) / 2
    asym_values = (da.values - da.values[:, ::-1, :]) / 2
    ds = xr.Dataset(
        {
            "sym": ([time_name, lat_name, lon_name], sym_values[:, nlat // 2 :, :]),
            "asym": ([time_name, lat_name, lon_name], asym_values[:, nlat // 2 :, :]),
        },
        coords={
            time_name: da[time_name],
            lat_name: da[lat_name][nlat // 2 :],
            lon_name: da[lon_name],
        },
    )

    # rolling construct → shape: (..., time, window)
    rolled = ds.rolling({time_name: nseg}, center=False).construct("time_in_segment")

    # keep only fully-formed windows, then stride by `step`
    # the first (win-1) entries are NaN because the window is not full yet
    rolled = rolled.isel({time_name: slice(nseg - 1, None, step)})

    # rename the (now sparse) time axis to 'segment' for clarity
    rolled = rolled.rename({time_name: "segment"})
    rolled["segment"] = np.arange(rolled.sizes["segment"])

    # make "time_in_segment" a proper timedelta index (0..win-1 samples)
    rolled = rolled.assign_coords(
        time_in_segment=(np.arange(nseg) / samples_per_day * 24).astype(
            "timedelta64[h]"
        )
    )

    # detrend along time axis
    linear_trend = rolled["sym"].polyfit(dim="time_in_segment", deg=1)
    rolled["sym"] = rolled["sym"] - xr.polyval(
        rolled["time_in_segment"], linear_trend["polyfit_coefficients"]
    )  # linear trend
    linear_trend = rolled["asym"].polyfit(dim="time_in_segment", deg=1)
    rolled["asym"] = rolled["asym"] - xr.polyval(
        rolled["time_in_segment"], linear_trend["polyfit_coefficients"]
    )  # linear trend

    # apply Hann window along "time_in_segment" axis
    hann_window = np.concatenate(
        (
            np.hanning(hann_width)[: hann_width // 2],
            np.ones(rolled.sizes["time_in_segment"] - hann_width),
            np.hanning(hann_width)[hann_width // 2 :],
        ),
        axis=0,
    )
    hann_window = xr.DataArray(
        hann_window,
        coords={"time_in_segment": rolled["time_in_segment"]},
        dims=["time_in_segment"],
    )
    windowed = rolled * hann_window
    windowed = windowed.transpose("segment", lat_name, "time_in_segment", lon_name)

    # fourier transform along time_in_segment and lon axes
    freq = fft.fftshift(fft.fftfreq(nseg, d=1 / samples_per_day))
    wavenum = -fft.fftshift(fft.fftfreq(nlon, d=1 / samples_per_lon))
    fft2d_sym = np.fft.fft2(windowed["sym"])
    fft2d_asym = np.fft.fft2(windowed["asym"])
    power_sym_mean = np.fft.fftshift(
        np.mean(np.abs(fft2d_sym) ** 2 / (nseg * nlon), axis=(0, 1))
    )
    power_asym_mean = np.fft.fftshift(
        np.mean(np.abs(fft2d_asym) ** 2 / (nseg * nlon), axis=(0, 1))
    )

    ds = xr.Dataset(
        {
            "power_sym_mean": (["freq", "wavenum"], power_sym_mean),
            "power_asym_mean": (["freq", "wavenum"], power_asym_mean),
        },
        coords={
            "freq": freq,
            "wavenum": wavenum,
        },
    )

    # smooth the power spectrum with a [1, 2, 1] filter along both axes
    weight = xr.DataArray([0.25, 0.5, 0.25], dims=["window"])
    ds["power_sym_mean"] = (
        ds["power_sym_mean"]
        .rolling(freq=3, center=True)
        .construct(freq="window")
        .dot(weight)
        .rolling(wavenum=3, center=True)
        .construct(wavenum="window")
        .dot(weight)
    )
    ds["power_asym_mean"] = (
        ds["power_asym_mean"]
        .rolling(freq=3, center=True)
        .construct(freq="window")
        .dot(weight)
        .rolling(wavenum=3, center=True)
        .construct(wavenum="window")
        .dot(weight)
    )
    power_smooth = (ds["power_sym_mean"].copy() + ds["power_asym_mean"].copy()) / 2
    for i in range(num_smooth):
        power_smooth = (
            power_smooth.rolling(freq=3, center=True)
            .construct(freq="window")
            .dot(weight)
        )
    for i in range(num_smooth):
        power_smooth = (
            power_smooth.rolling(wavenum=3, center=True)
            .construct(wavenum="window")
            .dot(weight)
        )
    ds["power_smooth"] = power_smooth
    ds["power_sym_ratio"] = ds["power_sym_mean"] / power_smooth
    ds["power_asym_ratio"] = ds["power_asym_mean"] / power_smooth

    return ds


def spectral_filtering(
    da,
    time_name="time",
    lon_name="lon",
    lat_name="lat",
    hann_days=5,
    wavenumber_band=(-20, -6),
    frequency_band=(1 / 5, 1 / 2.5),
    equivalent_depth_band=(8, 90),
):
    """
    Assuming `da` has a regular time, lat, lon grid, and lat dimension is symmetric about the equator.
    """
    dt = da[time_name][1] - da[time_name][0]  # assume regular time axis
    dlon = da[lon_name][1] - da[lon_name][0]  # assume regular lon axis
    samples_per_day = np.timedelta64(1, "D") / dt.values
    samples_per_lon = 360.0 / dlon.values

    nt = da.sizes[time_name]
    nlon = da.sizes[lon_name]
    nlat = da.sizes[lat_name]

    hann_width = int(hann_days * samples_per_day)

    # symmetric and asymmetric array
    da = da.transpose(time_name, lat_name, lon_name).sortby(lat_name)

    # detrend along time axis
    linear_trend = da.polyfit(dim=time_name, deg=1)
    da = da - xr.polyval(
        da[time_name], linear_trend["polyfit_coefficients"]
    )  # linear trend

    # apply Hann window along time_name axis
    hann_window = np.concatenate(
        (
            np.hanning(hann_width)[: hann_width // 2],
            np.ones(da.sizes[time_name] - hann_width),
            np.hanning(hann_width)[hann_width // 2 :],
        ),
        axis=0,
    )
    hann_window = xr.DataArray(
        hann_window,
        coords={time_name: da[time_name]},
        dims=[time_name],
    )
    windowed = da * hann_window
    windowed = windowed.transpose(lat_name, time_name, lon_name)

    # fourier transform along time_name and lon axes
    freq = fft.fftfreq(nt, d=1 / samples_per_day)
    wavenum = -fft.fftfreq(nlon, d=1 / samples_per_lon)
    fft2d_windowed = xr.DataArray(
        np.fft.fft2(windowed),
        coords={
            lat_name: windowed[lat_name],
            "freq": freq,
            "wavenum": wavenum,
        },
        dims=[lat_name, "freq", "wavenum"],
    )

    # filtering
    filtered = fft2d_windowed.where(
        (fft2d_windowed["wavenum"] >= wavenumber_band[0])
        & (fft2d_windowed["wavenum"] <= wavenumber_band[1])
        & (fft2d_windowed["freq"] >= frequency_band[0])
        & (fft2d_windowed["freq"] <= frequency_band[1]),
        0,
    )

    # inverse fourier transform
    ifft2d = xr.DataArray(
        np.fft.ifft2(filtered.values).real,
        coords={
            lat_name: windowed[lat_name],
            time_name: windowed[time_name],
            lon_name: windowed[lon_name],
        },
        dims=[lat_name, time_name, lon_name],
    ).transpose(time_name, lat_name, lon_name)

    return ifft2d
