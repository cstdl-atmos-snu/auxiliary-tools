import numpy as np
import xarray as xr
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
