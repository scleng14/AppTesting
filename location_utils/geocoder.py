from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="geo_locator_v1")
reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def get_address_from_coords(coords):
    """坐标反查地址"""
    try:
        location = reverse_geocode(coords, language='en')
        return location.address if location else None
    except Exception:
        return None
