"""Project WGS84 coordinates into a local metric coordinate system."""
import numpy as np
from pyproj import CRS, Transformer


def local_projection(lon, lat):
    center_lon, center_lat = float(np.median(lon)), float(np.median(lat))
    if not -80 <= center_lat <= 84:
        raise ValueError("Network is outside the supported UTM latitude range")
    zone = min(60, int((center_lon + 180) // 6) + 1)
    crs = CRS.from_epsg((32600 if center_lat >= 0 else 32700) + zone)
    return crs, Transformer.from_crs("EPSG:4326", crs, always_xy=True)
