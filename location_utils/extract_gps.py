from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def extract_gps_from_image(image: Image.Image):
    try:
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                for key in value:
                    decode = GPSTAGS.get(key, key)
                    gps_info[decode] = value[key]

        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = convert_to_degrees(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
            lon = convert_to_degrees(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))
            return lat, lon
        else:
            return None

    except Exception as e:
        return None


def convert_to_degrees(value, ref):
    d = float(value[0][0]) / float(value[0][1])
    m = float(value[1][0]) / float(value[1][1])
    s = float(value[2][0]) / float(value[2][1])

    degrees = d + (m / 60.0) + (s / 3600.0)
    if ref in ['S', 'W']:
        degrees = -degrees
    return degrees
