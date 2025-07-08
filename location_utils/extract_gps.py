from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

MAX_RETRIES = 2

def extract_gps(image_path):
    try:
        with Image.open(image_path) as img:
            exif = {
                TAGS.get(k): v
                for k, v in img._getexif().items()
                if k in TAGS
            }
            gps_info = exif.get('GPSInfo', {})
            return {
                GPSTAGS.get(k): v
                for k, v in gps_info.items()
                if k in GPSTAGS
            }
    except Exception as e:
        print(f"[EXIF ERROR] {str(e)}")
        return None

def convert_gps(gps):
    try:
        lat_data = gps['GPSLatitude']
        lon_data = gps['GPSLongitude']

        lat = lat_data[0] + lat_data[1]/60 + lat_data[2]/3600
        lon = lon_data[0] + lon_data[1]/60 + lon_data[2]/3600

        if gps['GPSLatitudeRef'] == 'S':
            lat = -lat
        if gps['GPSLongitudeRef'] == 'W':
            lon = -lon

        return round(lat, 6), round(lon, 6)
    except Exception as e:
        print(f"[GPS CONVERT ERROR] {str(e)}")
        return None

geolocator = Nominatim(user_agent="geo_locator_pro_v3")
reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=2)

def get_address_from_coords(coords):
    for _ in range(MAX_RETRIES):
        try:
            location = reverse_geocode(coords, language='en')
            return location.address if location else None
        except Exception as e:
            print(f"[GEOCODE RETRY {_+1}] {str(e)}")
            time.sleep(1)
    return None
