from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# 初始化 geopy 的地理编码器（用于反向地理编码）
geolocator = Nominatim(user_agent="location_app_gps")
reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def get_address_from_coords(coords):
    """
    使用纬度和经度坐标进行反向地理编码，返回地址信息。
    参数:
        coords: (lat, lon) 的元组
    返回:
        字符串地址 或 None
    """
    try:
        location = reverse_geocode(coords, language="en")
        return location.address if location else None
    except Exception:
        return None
