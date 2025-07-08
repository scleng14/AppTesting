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
    """将 DMS 坐标转换为十进制度
