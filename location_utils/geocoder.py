from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

# 创建 geolocator 实例（避免每次都创建新的）
geolocator = Nominatim(user_agent="emotion-location-app", timeout=10)

def reverse_geocode(coords):
    """
    根据 (lat, lon) 坐标获取人类可读的地理位置描述。
    参数:
        coords: tuple of (lat, lon)
    返回:
        地址字符串或 "Unknown"
    """
    try:
        location = geolocator.reverse(coords, language="en")
        if location and location.address:
            return location.address
        return "Unknown"
    except (GeocoderUnavailable, GeocoderTimedOut):
        return "Unknown"
