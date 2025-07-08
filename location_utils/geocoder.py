# geocoder.py
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

_geolocator = Nominatim(user_agent="location_detector")
_reverse = RateLimiter(_geolocator.reverse, min_delay_seconds=1)

def reverse_geocode(coords, language='en'):
    try:
        location = _reverse(coords, language=language)
        return location.address if location else None
    except Exception:
        return None
