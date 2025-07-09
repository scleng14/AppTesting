from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="emotion-location-app", timeout=10)
rate_limited_reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def reverse_geocode(coords):
    try:
        location = rate_limited_reverse(coords, language="en")
        if location and location.address:
            return location.address
        return "Unknown"
    except (GeocoderUnavailable, GeocoderTimedOut):
        return "Unknown"
    except Exception:
        return "Unknown"
