# extract_gps.py
from PIL import Image, ExifTags

def extract_gps(image: Image.Image):
    """从 PIL Image 中提取 GPS 原始信息"""
    try:
        if not hasattr(image, '_getexif') or image._getexif() is None:
            return None

        exif_data = image._getexif()
        gps_info = {}

        for key, val in exif_data.items():
            tag = ExifTags.TAGS.get(key)
            if tag == "GPSInfo":
                for t in val:
                    sub_tag = ExifTags.GPSTAGS.get(t)
                    gps_info[sub_tag] = val[t]

        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = _convert_to_degrees(gps_info["GPSLatitude"])
            lon = _convert_to_degrees(gps_info["GPSLongitude"])

            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat
            if gps_info.get("GPSLongitudeRef") == "W":
                lon = -lon

            return (lat, lon)
        return None
    except Exception:
        return None

def _convert_to_degrees(value):
    """将 GPS 度/分/秒 转换为十进制度"""
    try:
        d, m, s = value
        return d + m / 60.0 + s / 3600.0
    except Exception:
        return None

def extract_gps_from_image(pil_image):
    return extract_gps(pil_image)
