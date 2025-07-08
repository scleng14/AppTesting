from PIL import Image, ExifTags
from .geocoder import get_address_from_coords

def extract_gps_info(image_path):
    """提取图像中的 GPS 信息（如果存在）"""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None

        gps_data = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = ExifTags.GPSTAGS.get(t)
                    gps_data[sub_decoded] = value[t]
        return gps_data if gps_data else None
    except Exception:
        return None

def convert_to_decimal(degree, ref):
    """将 DMS 坐标转换为十进制度"""
    d, m, s = degree
    decimal = d + m / 60 + s / 3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def parse_coordinates(gps_info):
    """从 GPS 信息中提取经纬度坐标"""
    try:
        lat = convert_to_decimal(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
        lon = convert_to_decimal(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
        return round(lat, 6), round(lon, 6)
    except Exception:
        return None

def get_location(image_path):
    """提取图像的 GPS 位置并返回地址"""
    gps_info = extract_gps_info(image_path)
    if not gps_info:
        return None, "No GPS"

    coords = parse_coordinates(gps_info)
    if not coords:
        return None, "Invalid GPS"

    address = get_address_from_coords(coords)
    if address:
        return address, "GPS"
    return None, "Reverse Failed"
