from PIL.ExifTags import TAGS, GPSTAGS

def extract_gps(image):
    """从 PIL Image 中提取 GPS 数据"""
    try:
        if not hasattr(image, '_getexif'):
            return None
            
        exif = {
            TAGS.get(k): v 
            for k, v in image._getexif().items() 
            if k in TAGS
        }
        gps_info = exif.get('GPSInfo', {})
        return {
            GPSTAGS.get(k): v 
            for k, v in gps_info.items() 
            if k in GPSTAGS
        }
    except Exception:
        return None

def convert_gps(gps):
    """将 GPS 坐标转换为十进制"""
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
    except Exception:
        return None
