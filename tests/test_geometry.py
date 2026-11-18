from pyproj import Geod
from pathsnap_ml.geometry import local_projection


def test_metric_projection():
    crs,transform=local_projection([8,8.001],[49,49])
    x1,y1=transform.transform(8,49)
    x2,y2=transform.transform(8.001,49)
    geodesic=Geod(ellps="WGS84").inv(8,49,8.001,49)[2]
    assert crs.is_projected and abs(geodesic-abs(complex(x2-x1,y2-y1)))<0.1
