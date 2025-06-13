# metapi7_v6_flask_ready.py
import os
import serial
import time
import cv2
import numpy as np
from ultralytics import YOLO
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import piexif
import piexif.helper
from datetime import datetime
import logging
import json # Pour lire config_v6.json
import database_v6 as db # Utilise le module DB V6

# --- Configuration Globale ---
# Ports et Communication
SERIAL_PORT = '/dev/ttyUSB2' # Adaptez si nécessaire
BAUDRATE = 115200
GPS_TIMEOUT = 5
GPS_ACQUISITION_WAIT = 3
MIN_SPEED_FOR_COURSE_KNOTS = 0.2 # Vitesse minimale (en nœuds) pour considérer le cap GPS fiable

# Modèle IA et Détection
MODEL_NAME = "yolov8n.pt"

# Fichiers et Dossiers
BASE_DIR = '/home/acevik' # Assurez-vous que c'est correct
CONFIG_FILE = os.path.join(BASE_DIR, 'config_v6.json')
IMAGE_SAVE_DIR = os.path.join(BASE_DIR, 'captures')
DB_PATH = db.DB_PATH # Utilise le chemin défini dans database_v6

# Paramètres par Défaut (utilisés si config.json manque ou est invalide)
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_CONFIG = {
    "detection_confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
    "live_stream_enabled": False,
    "live_stream_port": 8001,
    "live_stream_resolution_width": 640,
    "live_stream_resolution_height": 480
}

# Capture Image
CAPTURE_TIMEOUT_MS = 700

# Paramètres Caméra (!!! CALIBRATION MANUELLE REQUISE !!!)
CAMERA_PARAMS = {
    'altitude': 5.0, # Altitude du drone au-dessus du sol (m)
    # --- MODIFIÉ ---
    'camera_yaw_relative_deg': 0.0, # Yaw de la caméra RELATIF à l'axe AVANT du drone (degrés). 0 = pointe vers l'avant.
    # --- FIN MODIF ---
    'pitch': 20.0, # Pitch de la caméra vers le bas (degrés). 0 = horizontal, 90 = nadir.
    'focal_length_px': 3571.43, # Distance focale équivalente en pixels
    'sensor_width_px': 1920, # Largeur du capteur en pixels
    'sensor_height_px': 1080, # Hauteur du capteur en pixels
    'make': "Raspberry Pi",
    'model': "Camera Module 3" # Adaptez
}

# --- Initialisation ---
console = Console()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Variables Globales pour Flask ---
loaded_config = None
yolo_model_global = None
is_initialized = False

# --- Fonctions Utilitaires (EXIF, etc.) ---
# _degrees_to_dms_rational, _create_gps_ifd, add_exif_metadata (PAS DE CHANGEMENTS ICI)
def _degrees_to_dms_rational(degree_float):
    """Convertit degrés décimaux en tuple DMS Rational pour EXIF."""
    if degree_float is None: return None
    try:
        sign = 1 if degree_float >= 0 else -1
        degree_float = abs(degree_float)
        degrees = int(degree_float)
        minutes_float = (degree_float - degrees) * 60
        minutes = int(minutes_float)
        seconds_float = (minutes_float - minutes) * 60
        seconds_num = int(seconds_float * 10000)
        seconds_den = 10000
        return ((degrees, 1), (minutes, 1), (seconds_num, seconds_den))
    except Exception as e:
        logger.error(f"Error converting degrees to DMS: {e}", exc_info=True)
        return None

def _create_gps_ifd(latitude, longitude, altitude, date_str, time_str):
    """Crée le dictionnaire GPS IFD pour piexif."""
    try:
        if latitude is None or longitude is None: return {}
        gps_ifd = {}
        lat_dms = _degrees_to_dms_rational(latitude)
        if lat_dms:
            gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = 'N' if latitude >= 0 else 'S'
            gps_ifd[piexif.GPSIFD.GPSLatitude] = lat_dms
        lon_dms = _degrees_to_dms_rational(longitude)
        if lon_dms:
            gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = 'E' if longitude >= 0 else 'W'
            gps_ifd[piexif.GPSIFD.GPSLongitude] = lon_dms

        gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0
        alt_num = int(abs(altitude or 0.0) * 100)
        alt_den = 100
        gps_ifd[piexif.GPSIFD.GPSAltitude] = (alt_num, alt_den)

        if date_str and time_str:
            try:
                time_part = time_str.split('.')[0]
                dt_obj = datetime.strptime(f"{date_str}{time_part}", "%d%m%y%H%M%S")
                gps_ifd[piexif.GPSIFD.GPSDateStamp] = dt_obj.strftime("%Y:%m:%d")
                gps_ifd[piexif.GPSIFD.GPSTimeStamp] = [(dt_obj.hour, 1), (dt_obj.minute, 1), (dt_obj.second, 1)]
            except ValueError as dt_err:
                logger.warning(f"EXIF Warning: Could not parse GPS date/time: {dt_err} (date='{date_str}', time='{time_str}')")
        return gps_ifd
    except Exception as e:
        logger.error(f"Error creating GPS IFD: {e}", exc_info=True)
        return {}

def add_exif_metadata(image_path, gps_ifd=None, user_comment="", camera_make="", camera_model=""):
    """Ajoute/Met à jour les métadonnées EXIF d'une image."""
    logger.debug(f"Adding EXIF metadata to {os.path.basename(image_path)}")
    if not os.path.exists(image_path):
        logger.error(f"EXIF Error: Image file not found at path: {image_path}")
        return
    try:
        try: exif_dict = piexif.load(image_path)
        except piexif.InvalidImageDataError: logger.debug(f"No valid EXIF found in {os.path.basename(image_path)}"); exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        except Exception as load_err: logger.warning(f"EXIF Warning: Could not load existing EXIF: {load_err}"); exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        if gps_ifd: exif_dict["GPS"] = gps_ifd
        now_str = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        exif_dict["0th"][piexif.ImageIFD.DateTime] = now_str
        if camera_make: exif_dict["0th"][piexif.ImageIFD.Make] = camera_make
        if camera_model: exif_dict["0th"][piexif.ImageIFD.Model] = camera_model
        exif_dict["0th"][piexif.ImageIFD.Software] = "Metapi Detection System v6.1 (Flask+Heading)" # Version Update

        if "Exif" not in exif_dict: exif_dict["Exif"] = {}
        if piexif.ExifIFD.DateTimeOriginal not in exif_dict["Exif"]: exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = now_str
        if piexif.ExifIFD.DateTimeDigitized not in exif_dict["Exif"]: exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = now_str
        if user_comment:
            try: exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment, encoding="unicode")
            except Exception as comment_err: logger.warning(f"Could not encode user comment for EXIF: {comment_err}")

        try:
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, image_path)
            logger.info(f"Successfully updated EXIF for {os.path.basename(image_path)}")
        except Exception as dump_err: logger.error(f"Error dumping or inserting EXIF data: {dump_err}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error writing EXIF metadata: {e}", exc_info=True)

# --- Fonction Load Config V6 --- (PAS DE CHANGEMENTS ICI)
def load_config():
    """Charge la configuration depuis config_v6.json, avec valeurs par défaut."""
    global loaded_config
    logger.info(f"Loading configuration from {CONFIG_FILE}")
    config = DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            loaded_config_file = json.load(f)
            config.update(loaded_config_file)
            logger.info(f"Config file loaded and merged with defaults.")
    except FileNotFoundError:
        logger.warning(f"Config file {CONFIG_FILE} not found. Using default configuration: {config}")
    except json.JSONDecodeError as e:
        logger.error(f"Config Error: Failed to decode JSON from {CONFIG_FILE}: {e}. Using default configuration: {config}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}. Using default configuration: {config}", exc_info=True)

    try:
        threshold = float(config["detection_confidence_threshold"])
        if not (0.0 <= threshold <= 1.0):
            logger.warning(f"Invalid threshold '{threshold}' in config, reverting to default {DEFAULT_CONFIG['detection_confidence_threshold']:.2f}")
            config["detection_confidence_threshold"] = DEFAULT_CONFIG['detection_confidence_threshold']
        else:
             config["detection_confidence_threshold"] = threshold
    except (ValueError, TypeError, KeyError):
        logger.warning(f"Invalid/Missing 'detection_confidence_threshold' in config, using default {DEFAULT_CONFIG['detection_confidence_threshold']:.2f}")
        config["detection_confidence_threshold"] = DEFAULT_CONFIG['detection_confidence_threshold']

    logger.info(f"Final configuration applied: {config}")
    loaded_config = config
    return loaded_config

# --- Fonctions Principales (GPS, Capture, Estimate) ---
# get_gps_data, capture_image (PAS DE CHANGEMENTS ICI)
def get_gps_data():
    """Récupère les données GPS via port série (AT+CGPSINFO)."""
    logger.debug("Attempting to get GPS data...")
    latitude, longitude, date_str, time_str, altitude, speed_knots, course = None, None, None, None, None, None, None
    raw_response = ""
    try:
        with serial.Serial(SERIAL_PORT, baudrate=BAUDRATE, timeout=GPS_TIMEOUT) as ser:
            logger.debug(f"Serial port {SERIAL_PORT} opened.")
            ser.write(b'AT\r\n'); time.sleep(0.2)
            at_response = ser.read(ser.inWaiting()).decode('utf-8', errors='ignore')
            logger.debug(f"AT response: {at_response.strip()}")
            if "OK" not in at_response:
                logger.warning("Did not receive OK from modem. Trying GPS commands anyway.")

            ser.reset_input_buffer()
            ser.write(b'AT+CGPS?\r\n'); time.sleep(0.5)
            response_check = ser.read(ser.inWaiting()).decode('utf-8', errors='ignore')
            logger.debug(f"AT+CGPS? response: {response_check.strip()}")
            if "+CGPS: 0" in response_check:
                logger.warning("GPS off, enabling (AT+CGPS=1)...")
                ser.write(b'AT+CGPS=1\r\n'); time.sleep(3) # Give it time to potentially acquire
                ser.reset_input_buffer()
                ser.write(b'AT+CGPS?\r\n'); time.sleep(0.5)
                response_check_after_enable = ser.read(ser.inWaiting()).decode('utf-8', errors='ignore')
                logger.debug(f"AT+CGPS? response after enable: {response_check_after_enable.strip()}")

            logger.debug("Sending AT+CGPSINFO..."); ser.reset_input_buffer()
            ser.write(b'AT+CGPSINFO\r\n'); time.sleep(GPS_ACQUISITION_WAIT)
            raw_response = ser.read(ser.inWaiting()).decode('utf-8', errors='ignore')
            logger.debug(f"AT+CGPSINFO raw response: {raw_response.strip()}")

        if '+CGPSINFO:' in raw_response:
            info_part = ""
            for line in raw_response.splitlines():
                line_strip = line.strip()
                if line_strip.startswith('+CGPSINFO:'):
                    info_part = line_strip.split('+CGPSINFO: ', 1)[1]
                    break
            if not info_part:
                logger.error("GPS Error: Empty CGPSINFO info part found in response.");
                return None, None, None, None, None, None, None, raw_response

            parts = info_part.split(',')
            if len(parts) >= 6 and parts[0] and parts[2]:
                try:
                    lat_raw = parts[0]; lat_deg = float(lat_raw[:2]); lat_min = float(lat_raw[2:])
                    latitude = lat_deg + lat_min / 60.0
                    if parts[1] == 'S': latitude = -latitude

                    lon_raw = parts[2]; lon_deg = float(lon_raw[:3]); lon_min = float(lon_raw[3:])
                    longitude = lon_deg + lon_min / 60.0
                    if parts[3] == 'W': longitude = -longitude

                    date_str = parts[4] if len(parts) > 4 else None
                    time_str = parts[5] if len(parts) > 5 else None
                    altitude = float(parts[6]) if len(parts) > 6 and parts[6] else 0.0
                    speed_knots = float(parts[7]) if len(parts) > 7 and parts[7] else 0.0
                    # --- Course Handling ---
                    course = None # Default to None
                    if len(parts) > 8 and parts[8]:
                         try:
                              course_raw = float(parts[8])
                              if speed_knots >= MIN_SPEED_FOR_COURSE_KNOTS: # Use course only if speed is sufficient
                                   course = course_raw
                                   logger.debug(f"Valid course received: {course:.1f} deg (Speed: {speed_knots:.1f} knots)")
                              else:
                                   logger.debug(f"Course ({course_raw:.1f} deg) ignored due to low speed ({speed_knots:.1f} < {MIN_SPEED_FOR_COURSE_KNOTS} knots)")
                         except ValueError:
                              logger.warning(f"Could not parse course value: {parts[8]}")
                    else:
                         logger.debug("Course data not available in GPS output.")
                    # --- End Course Handling ---

                    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                        logger.warning(f"GPS Warning: Invalid coordinates parsed: Lat={latitude}, Lon={longitude}. Raw: '{info_part}'");
                        return None, None, None, None, None, None, None, raw_response
                    if not date_str or len(date_str) != 6 or not time_str or '.' not in time_str:
                         logger.warning(f"GPS Warning: Invalid date/time format: Date='{date_str}', Time='{time_str}'. Raw: '{info_part}'");

                    log_course_str = f", Course={course:.1f}deg" if course is not None else ", Course=N/A"
                    logger.info(f"GPS Fix acquired: Lat={latitude:.6f}, Lon={longitude:.6f}, Alt={altitude:.1f}m, Speed={speed_knots:.1f}kn{log_course_str}")
                    return latitude, longitude, date_str, time_str, altitude, speed_knots, course, raw_response
                except (ValueError, IndexError, TypeError) as e:
                    logger.error(f"GPS Parse Error: {e} - Data: '{info_part}'", exc_info=True);
                    return None, None, None, None, None, None, None, raw_response
            else:
                 logger.info(f"GPS Info: No valid fix (empty or malformed data). Data: '{info_part}'");
                 return None, None, None, None, None, None, None, raw_response
        elif "+CGPS: 1,1" in raw_response or "+CGPS: 1,0" in raw_response or "OK" in raw_response:
             logger.info(f"GPS Info: Module active but no '+CGPSINFO:' line received yet. Raw: '{raw_response.strip()}'")
             return None, None, None, None, None, None, None, raw_response
        else:
             logger.warning(f"GPS Warning: Unexpected response or no '+CGPSINFO:' line. Raw: '{raw_response.strip()}'");
             return None, None, None, None, None, None, None, raw_response
    except serial.SerialException as e:
        logger.error(f"GPS Serial Error on {SERIAL_PORT}: {e}", exc_info=True);
        return None, None, None, None, None, None, None, f"Serial Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in get_gps_data: {e}", exc_info=True);
        return None, None, None, None, None, None, None, f"Unexpected Error: {e}"

def capture_image(image_path):
    """Capture une image avec libcamera-still."""
    logger.info(f"Capturing image to {os.path.basename(image_path)}...")
    img_dir = os.path.dirname(image_path)
    if not os.path.exists(img_dir):
        try: os.makedirs(img_dir); logger.info(f"Created directory: {img_dir}")
        except OSError as e: logger.error(f"Capture Error: Failed to create directory {img_dir}: {e}"); return False

    if os.system("which libcamera-still > /dev/null 2>&1") != 0:
         logger.error("Capture Error: 'libcamera-still' command not found in PATH."); return False

    command = (
        f"libcamera-still -o '{image_path}' "
        f"--width {CAMERA_PARAMS['sensor_width_px']} "
        f"--height {CAMERA_PARAMS['sensor_height_px']} "
        f"--timeout {CAPTURE_TIMEOUT_MS} "
        f"--nopreview"
    )
    logger.debug(f"Executing: {command}")
    try:
        capture_status = os.system(command);
        exit_code = os.waitstatus_to_exitcode(capture_status) if os.WIFEXITED(capture_status) else -1
    except Exception as e:
        logger.error(f"Exception during capture command execution: {e}", exc_info=True); return False

    if exit_code == 0:
        if os.path.exists(image_path):
            try:
                if os.path.getsize(image_path) > 1000:
                    logger.info(f"Image captured successfully: {os.path.basename(image_path)}")
                    return True
                else:
                    logger.error(f"Image capture failed! File exists but is too small ({os.path.getsize(image_path)} bytes): {os.path.basename(image_path)}")
                    return False
            except OSError as stat_err:
                 logger.error(f"Error checking size of captured file {image_path}: {stat_err}")
                 return False
        else:
            logger.error(f"Image capture failed! Exit code was 0, but file does not exist: {os.path.basename(image_path)}")
            return False
    else:
        logger.error(f"Image capture failed! libcamera-still exited with code: {exit_code}.")
        if os.path.exists(image_path):
             logger.warning(f"Removing potentially corrupted file due to non-zero exit code: {image_path}")
             try: os.remove(image_path)
             except OSError as rm_err: logger.error(f"Could not remove failed capture file {image_path}: {rm_err}")
        return False

# --- MODIFIÉ ---
def estimate_person_location(module_lat, module_lon, drone_heading_deg, # Ajout du cap drone
                             camera_params, # Utilise le dict
                             x_center_norm, y_center_norm):
    """Estime la position GPS de la personne détectée en utilisant le cap du drone."""

    # Extrait les paramètres nécessaires
    cam_alt = camera_params['altitude']
    cam_pitch_deg = camera_params['pitch']
    cam_yaw_relative_deg = camera_params['camera_yaw_relative_deg'] # Yaw relatif à l'avant du drone
    focal_px = camera_params['focal_length_px']
    sensor_w_px = camera_params['sensor_width_px']
    sensor_h_px = camera_params['sensor_height_px']

    logger.debug(f"Estimating location: Det@({x_center_norm:.3f}, {y_center_norm:.3f}), DroneHdg={drone_heading_deg}, CamYawRel={cam_yaw_relative_deg}")

    # Vérifie si le cap du drone est disponible et fiable
    if drone_heading_deg is None:
        logger.warning("Estimation Warning: Drone heading is None (likely stationary or GPS issue). Cannot reliably estimate absolute position.")
        return None, None, None # Impossible de calculer l'azimut absolu

    try:
        # Angle horizontal de la détection RELATIF AU CENTRE de la caméra (radians)
        angle_x_relative_cam_rad = np.arctan(((x_center_norm - 0.5) * sensor_w_px) / focal_px)
        # Angle vertical de la détection RELATIF AU CENTRE de la caméra (radians, positif vers le bas)
        angle_y_image_rad = np.arctan(((y_center_norm - 0.5) * sensor_h_px) / focal_px)

        # Angle vertical absolu vers le bas depuis l'horizon (radians)
        true_vertical_angle_down_rad = np.radians(cam_pitch_deg) + angle_y_image_rad

        # Validation de l'angle vertical
        if true_vertical_angle_down_rad <= np.radians(1.0):
            logger.warning(f"Estimation Warning: Vertical angle ({np.degrees(true_vertical_angle_down_rad):.1f}°) too low.")
            return None, None, None
        if true_vertical_angle_down_rad >= np.radians(89.0):
            logger.warning(f"Estimation Warning: Vertical angle ({np.degrees(true_vertical_angle_down_rad):.1f}°) very high.")

        # Calcul de la distance horizontale au sol
        distance_horizontal_m = cam_alt / np.tan(true_vertical_angle_down_rad)

        if distance_horizontal_m < 0:
             logger.warning(f"Estimation Warning: Negative horizontal distance ({distance_horizontal_m:.1f}m).")
             return None, None, None
        if distance_horizontal_m > 5000: # Limite pratique
             logger.warning(f"Estimation Warning: Calculated large distance ({distance_horizontal_m:.1f}m).")

        # Calcul de l'Azimut ABSOLU (relatif au Nord) vers la détection
        # Azimut = Cap_Drone + Yaw_Caméra_Relatif + Angle_Détection_Relatif_Caméra
        detection_angle_relative_cam_deg = np.degrees(angle_x_relative_cam_rad)
        absolute_azimuth_deg = (drone_heading_deg + cam_yaw_relative_deg + detection_angle_relative_cam_deg) % 360
        azimuth_rad = np.radians(absolute_azimuth_deg) # Conversion en radians pour sin/cos

        logger.debug(f"Calculated Azimuth: DroneHdg({drone_heading_deg:.1f}) + CamYawRel({cam_yaw_relative_deg:.1f}) + DetAngRel({detection_angle_relative_cam_deg:.1f}) = AbsAzimuth({absolute_azimuth_deg:.1f})")

        # Calcul des nouvelles coordonnées GPS (approximation terre plate pour courtes distances)
        earth_radius_m = 6371000.0
        lat1_rad = np.radians(module_lat)
        lon1_rad = np.radians(module_lon)

        delta_lat_rad = (distance_horizontal_m * np.cos(azimuth_rad)) / earth_radius_m
        # Correction pour la longitude basée sur la latitude
        delta_lon_rad = (distance_horizontal_m * np.sin(azimuth_rad)) / (earth_radius_m * np.cos(lat1_rad))

        estimated_person_lat_rad = lat1_rad + delta_lat_rad
        estimated_person_lon_rad = lon1_rad + delta_lon_rad

        estimated_person_lat = np.degrees(estimated_person_lat_rad)
        estimated_person_lon = np.degrees(estimated_person_lon_rad)

        # Validation finale
        if not (-90 <= estimated_person_lat <= 90 and -180 <= estimated_person_lon <= 180):
            logger.error(f"Estimation Error: Coordinates out of bounds ({estimated_person_lat:.6f}, {estimated_person_lon:.6f}).");
            return None, None, None

        logger.info(f"Estimated location: Lat={estimated_person_lat:.6f}, Lon={estimated_person_lon:.6f} (Dist: {distance_horizontal_m:.1f}m @ Azm: {absolute_azimuth_deg:.1f}°)")
        return estimated_person_lat, estimated_person_lon, distance_horizontal_m

    except ZeroDivisionError:
        logger.error("Estimation Error: Division by zero.", exc_info=True);
        return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error during location estimation: {e}", exc_info=True);
        return None, None, None
# --- FIN MODIF ---

# --- MODIFIÉ ---
def detect_human(yolo_model, image_path, module_gps_data, camera_params, config):
    """Détecte, estime (utilisant cap drone), annote, sauvegarde image et enregistre dans la DB."""
    if yolo_model is None: logger.error("Detection Error: YOLO model is None."); return 0, []
    if config is None: logger.error("Detection Error: Configuration is None."); return 0, []
    current_confidence_threshold = config.get("detection_confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)

    logger.info(f"Processing image: {os.path.basename(image_path)} with threshold {current_confidence_threshold:.2f}")
    detections_for_summary = []
    user_comment_lines = []
    saved_detections_count = 0

    # Extrait les données GPS (incluant vitesse et cap)
    module_lat, module_lon, date_str, time_str, module_alt, speed_knots, drone_course, gps_raw = module_gps_data
    gps_fixed = module_lat is not None and module_lon is not None
    drone_heading_reliable = drone_course is not None # Le cap est considéré fiable s'il n'est pas None (déjà filtré par vitesse dans get_gps_data)

    module_fix_time_str = None
    if gps_fixed and date_str and time_str:
         try:
             time_part = time_str.split('.')[0]
             dt_obj = datetime.strptime(f"{date_str}{time_part}", "%d%m%y%H%M%S")
             module_fix_time_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
         except ValueError:
             logger.warning(f"Could not parse module fix time: date='{date_str}', time='{time_str}'")
             module_fix_time_str = f"{date_str}_{time_str}"

    # Prépare le commentaire EXIF initial
    gps_info_str = f"Lat={module_lat:.6f}, Lon={module_lon:.6f}, Alt={module_alt:.1f}m" if gps_fixed else "No Fix"
    heading_info_str = f"Hdg={drone_course:.1f}deg" if drone_heading_reliable else "Hdg=N/A"
    speed_info_str = f"Spd={speed_knots:.1f}kn" if speed_knots is not None else "Spd=N/A"
    user_comment_lines.append(f"Module: {gps_info_str}, {heading_info_str}, {speed_info_str} @ {module_fix_time_str or '?'} UTC")
    if not gps_fixed: user_comment_lines.append(f"(Raw: {gps_raw.strip()})")

    gps_ifd_module = _create_gps_ifd(module_lat, module_lon, module_alt, date_str, time_str)

    # Charge l'image
    try:
        if not os.path.exists(image_path): logger.error(f"Detection Error: Image file does not exist: {image_path}"); return 0, []
        image = cv2.imread(image_path)
        if image is None: logger.error(f"Detection Error: Failed to load image {os.path.basename(image_path)}"); return 0, []
        img_h, img_w, _ = image.shape
        logger.debug(f"Image {os.path.basename(image_path)} loaded ({img_w}x{img_h}).")
        # Vérification résolution vs params
        if camera_params.get('sensor_width_px') != img_w or camera_params.get('sensor_height_px') != img_h:
            logger.warning(f"Image resolution ({img_w}x{img_h}) differs from CAMERA_PARAMS. Estimation accuracy may be affected.")
    except Exception as e: logger.error(f"Detection Error: Failed loading image {os.path.basename(image_path)}: {e}", exc_info=True); return 0, []

    # Exécute la détection YOLO
    logger.debug(f"Running YOLO detection (threshold: {current_confidence_threshold:.2f})...")
    try:
        results = yolo_model(image_path, verbose=False, conf=current_confidence_threshold, classes=[0]) # Personne seulement
    except Exception as e: logger.error(f"YOLO detection failed for {image_path}: {e}", exc_info=True); return 0, []

    person_count = 0; image_annotated = False; detections_found_total = 0
    processed_results = []

    for r in results:
        detections_found_total += len(r.boxes)
        for box in r.boxes: # Devrait contenir uniquement des personnes
            person_count += 1; conf = float(box.conf[0])
            logger.debug(f"Person {person_count} detected with confidence {conf:.3f}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center_px = (x1 + x2) / 2; y_center_px = (y1 + y2) / 2
            x_center_norm = x_center_px / img_w; y_center_norm = y_center_px / img_h
            bbox_width = x2-x1; bbox_height = y2-y1

            estimated_lat, estimated_lon, estimated_dist_m, map_link = None, None, None, ""
            estimation_status = "NoGPSFix" # Default status

            if gps_fixed:
                if drone_heading_reliable:
                    # Tente l'estimation en passant le cap du drone
                    # Vérifie que tous les params requis sont présents
                    required_params = ['altitude', 'camera_yaw_relative_deg', 'pitch', 'focal_length_px', 'sensor_width_px', 'sensor_height_px']
                    if all(p in camera_params for p in required_params):
                         estimated_lat, estimated_lon, estimated_dist_m = estimate_person_location(
                             module_lat, module_lon, drone_course, # Passe le cap du drone
                             camera_params, # Passe le dict complet
                             x_center_norm, y_center_norm )
                         if estimated_lat is not None: estimation_status = "Estimated"
                         else: estimation_status = "EstimFail_Calc" # Echec pendant calcul
                    else:
                         logger.warning(f"Cannot estimate location: Missing required CAMERA_PARAMS: {required_params}")
                         estimation_status = "EstimFail_Params" # Paramètres manquants
                else:
                    estimation_status = "EstimFail_NoHdg" # GPS fixe mais pas de cap fiable
            # else: estimation_status reste "NoGPSFix"

            # Met à jour commentaire EXIF et lien maps basé sur le statut
            comment = f"P{person_count}: Conf={conf*100:.1f}% Status={estimation_status}"
            if estimation_status == "Estimated":
                map_link = f"https://www.google.com/maps/search/?api=1&query={estimated_lat:.8f},{estimated_lon:.8f}"
                comment += f" EstLoc=({estimated_lat:.6f},{estimated_lon:.6f}) Dist={estimated_dist_m:.1f}m"
            user_comment_lines.append(comment)

            # Prépare données pour DB
            detection_data_db = {
                'image_filename': os.path.basename(image_path),
                'module_lat': module_lat, 'module_lon': module_lon, 'module_alt': module_alt,
                'module_fix_time': module_fix_time_str, 'drone_heading': drone_course, 'drone_speed_knots': speed_knots, # Ajout infos drone
                'person_id_in_image': person_count, 'confidence': conf,
                'bbox_x1': x1, 'bbox_y1': y1, 'bbox_x2': x2, 'bbox_y2': y2,
                'bbox_center_x_px': int(x_center_px), 'bbox_center_y_px': int(y_center_px),
                'estimated_lat': estimated_lat, 'estimated_lon': estimated_lon,
                'estimated_distance_m': estimated_dist_m, 'map_link': map_link,
                'estimation_status': estimation_status # Ajout statut estimation
            }
            last_id = db.save_detection(detection_data_db)
            if last_id:
                 saved_detections_count += 1; detection_data_db['db_id'] = last_id
            else: detection_data_db['db_id'] = None
            processed_results.append(detection_data_db)

            # Dessine sur l'image
            label = f"P{person_count} ({conf*100:.0f}%)"
            color = (0, 0, 255) # Rouge (No GPS) par défaut
            if estimation_status == "Estimated":
                label += f" Est@{estimated_dist_m:.0f}m"
                color = (0, 255, 0) # Vert (Succès)
            elif estimation_status.startswith("EstimFail"):
                 label += f" {estimation_status.split('_')[-1]}" # Affiche la raison de l'échec
                 color = (0, 165, 255) # Orange (Echec estim mais GPS OK)
            # else: label += " NoGPS", color reste rouge

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + bbox_height + 15
            cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            image_annotated = True

    # Log global pour l'image
    if detections_found_total == 0: logger.info(f"No objects detected in {os.path.basename(image_path)}.")
    elif person_count == 0: logger.info(f"Detected {detections_found_total} objects, but no 'person' in {os.path.basename(image_path)}.")
    else: logger.info(f"Detected {person_count} person(s) in {os.path.basename(image_path)}.")

    # Sauvegarde image annotée
    if image_annotated:
        try:
            logger.debug(f"Saving annotated image: {os.path.basename(image_path)}")
            cv2.imwrite(image_path, image) # Écrase l'original
            logger.info(f"Annotated image saved: {os.path.basename(image_path)}")
        except Exception as e:
             logger.error(f"Failed to save annotated image {os.path.basename(image_path)}: {e}", exc_info=True)

    # Ajoute métadonnées EXIF
    final_user_comment = "\n".join(user_comment_lines)
    try:
        add_exif_metadata(image_path, gps_ifd=gps_ifd_module, user_comment=final_user_comment,
                          camera_make=camera_params.get('make', ''), camera_model=camera_params.get('model', ''))
    except Exception as exif_err:
         logger.error(f"Failed to add EXIF metadata to {os.path.basename(image_path)}: {exif_err}", exc_info=True)

    # Affiche tableau résumé Rich
    if processed_results:
        console.print(Panel(f"[bold]Detections Summary ({os.path.basename(image_path)})[/]", expand=False))
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", width=3); table.add_column("DB_ID", width=5); table.add_column("Conf", style="green")
        table.add_column("Est. Status", style="cyan"); table.add_column("Est. Pos / Dist", style="yellow"); table.add_column("Link", style="blue", no_wrap=True)
        for det in processed_results:
            pos_str = "[dim]N/A[/]"; link_str = "[dim]N/A[/]"
            status_color = "red"
            if det['estimation_status'] == "Estimated":
                dist_str = f"{det['estimated_distance_m']:.1f}m" if det['estimated_distance_m'] is not None else "?m"
                pos_str = f"{det['estimated_lat']:.6f},{det['estimated_lon']:.6f} ({dist_str})"
                link_str = f"[link={det['map_link']}]View[/link]" if det['map_link'] else "[dim]N/A[/dim]"
                status_color = "green"
            elif det['estimation_status'].startswith("EstimFail"):
                pos_str = "[orange3]Failed[/]"
                status_color = "orange3"
            # else: status NoGPSFix, pos_str N/A, color red

            table.add_row(
                 str(det['person_id_in_image']), str(det['db_id'] or '-'), f"{det['confidence']*100:.1f}%",
                 f"[{status_color}]{det['estimation_status']}[/]", pos_str, link_str )
        console.print(table)

    return saved_detections_count, processed_results
# --- FIN MODIF ---


# --- Initialisation Globale --- (PAS DE CHANGEMENTS ICI)
def initialize_system():
    """Charge la config, initialise la DB, et charge le modèle YOLO."""
    global loaded_config, yolo_model_global, is_initialized
    if is_initialized: logger.info("System already initialized."); return True

    logger.info("="*30); logger.info("Initializing Metapi System..."); logger.info("="*30)
    loaded_config = load_config()
    console.print(f"[info]Confidence Threshold: {loaded_config.get('detection_confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD)*100:.0f}%[/]")
    try:
        logger.info("Initializing database via db module...")
        db.init_db()
        logger.info("Database initialization check complete.")
        console.print(f"[green]✓[/] Database initialized ('{DB_PATH}').")
    except Exception as db_init_err:
        logger.critical(f"FATAL: DB init failed: {db_init_err}", exc_info=True); console.print(f"[bold red]FATAL: DB init failed.[/]"); return False
    try:
        logger.info(f"Loading YOLO model '{MODEL_NAME}'...")
        console.print(f"[blue]Loading YOLO model '{MODEL_NAME}'...[/]")
        model = YOLO(MODEL_NAME)
        yolo_model_global = model
        console.print(f"[green]✓[/] YOLO model loaded.")
        logger.info("YOLO model loaded.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to load YOLO model '{MODEL_NAME}': {e}", exc_info=True)
        console.print(f"[bold red]ERREUR CRITIQUE:[/bold red] Impossible de charger le modèle YOLO '{MODEL_NAME}'. Arrêt.")
        yolo_model_global = None; return False
    try:
        if not os.path.isdir(IMAGE_SAVE_DIR):
             logger.warning(f"Image save directory {IMAGE_SAVE_DIR} does not exist. Attempting to create.")
             os.makedirs(IMAGE_SAVE_DIR); logger.info(f"Created image save directory: {IMAGE_SAVE_DIR}")
        else: logger.info(f"Image save directory exists: {IMAGE_SAVE_DIR}")
    except Exception as dir_err:
        logger.error(f"Error checking or creating image save directory {IMAGE_SAVE_DIR}: {dir_err}", exc_info=True)

    is_initialized = True
    logger.info("="*30); logger.info("Metapi System Initialized Successfully."); logger.info("="*30)
    return True

# --- Exécuter un Cycle de Détection --- (PAS DE CHANGEMENTS ICI)
def run_single_detection_cycle():
    """Exécute une séquence complète: GPS -> Capture -> Détection."""
    global loaded_config, yolo_model_global
    if not is_initialized: logger.error("System not initialized."); return {"success": False, "error": "System not initialized", "details": None}
    if yolo_model_global is None: logger.error("YOLO model not loaded."); return {"success": False, "error": "YOLO model not loaded", "details": None}
    if loaded_config is None: logger.error("Configuration not loaded."); return {"success": False, "error": "Configuration not loaded", "details": None}

    logger.info("--- Starting Single Detection Cycle ---")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S"); image_filename = f"capture_{timestamp_str}.jpg"
    image_path = os.path.join(IMAGE_SAVE_DIR, image_filename); logger.debug(f"Target image path: {image_path}")

    # 1. Obtenir données GPS (inclut maintenant le cap filtré)
    gps_data = get_gps_data()
    gps_fixed = gps_data[0] is not None
    drone_course = gps_data[6] # Peut être None
    gps_log_str = f"[green]🛰️ GPS Fix:[/green] {gps_data[0]:.6f}, {gps_data[1]:.6f}" if gps_fixed else f"[red]🛰️ No GPS Fix[/red]"
    heading_log_str = f", [cyan]Hdg:{drone_course:.1f}°[/]" if drone_course is not None else ", [yellow]Hdg:N/A[/]"
    console.print(f"{gps_log_str}{heading_log_str if gps_fixed else ''}")
    if not gps_fixed: console.print(f"[dim](Raw: {gps_data[7].strip()})[/]")

    # 2. Capturer image
    capture_success = capture_image(image_path)

    # Prépare le dictionnaire de résultats (ajout cap)
    result_details = {
         "timestamp": timestamp_str, "image_filename": image_filename, "image_full_path": image_path,
         "gps_fixed": gps_fixed, "module_lat": gps_data[0], "module_lon": gps_data[1],
         "module_alt": gps_data[4], "drone_speed_knots": gps_data[5], "drone_heading": drone_course, # Ajout Vitesse/Cap
         "capture_success": capture_success, "detections_saved": 0, "detected_persons": []
    }

    if capture_success:
        try:
            # 3. Détecter humains (utilise maintenant le cap pour l'estimation)
            saved_count, processed_results = detect_human(
                yolo_model_global, image_path, gps_data, CAMERA_PARAMS, loaded_config
            )
            result_details["detections_saved"] = saved_count
            result_details["detected_persons"] = processed_results
            logger.info(f"Detection cycle complete for {image_filename}. Saved {saved_count} detections.")
            return {"success": True, "error": None, "details": result_details}
        except Exception as e:
            logger.error(f"Error during detection phase for {image_filename}: {e}", exc_info=True)
            console.print(f"[bold red]Error in detection phase. See logs.[/]")
            return {"success": False, "error": f"Detection phase failed: {e}", "details": result_details}
    else:
        logger.warning(f"Image capture failed. Skipping detection for this cycle.")
        return {"success": False, "error": "Image capture failed", "details": result_details}


# --- Fonction main V6 --- (MODIFIÉE pour afficher le Yaw Relatif)
def main_loop(num_captures=3, capture_delay=5):
    """Fonction principale en boucle pour le script autonome V6."""
    console.print(Panel.fit("[bold green]Système Détection & Localisation V6.1 (avec Cap Drone)[/]", subtitle="[italic]Mode Boucle Autonome[/]"))

    if not initialize_system():
        logger.critical("System initialization failed. Exiting main loop."); return

    # --- MODIFIÉ : Afficher le Yaw Relatif ---
    console.print(Panel(
        f"- Altitude: {CAMERA_PARAMS['altitude']}m, Cam Yaw Relatif: {CAMERA_PARAMS['camera_yaw_relative_deg']}°, Pitch: {CAMERA_PARAMS['pitch']}°\n"
        f"- Focale: {CAMERA_PARAMS['focal_length_px']}px, Résolution: {CAMERA_PARAMS['sensor_width_px']}x{CAMERA_PARAMS['sensor_height_px']}",
        title="[yellow]Params Caméra (Calibration Manuelle)[/]", border_style="yellow", expand=False ))
    # --- FIN MODIF ---

    total_saved_loop = 0
    logger.info(f"Starting detection loop: {num_captures} cycles, {capture_delay}s delay.")
    for i in range(num_captures):
        cycle_num = i + 1; console.rule(f"[bold cyan]Cycle {cycle_num}/{num_captures}[/]"); logger.info(f"--- Loop Cycle {cycle_num}/{num_captures} ---")
        cycle_result = run_single_detection_cycle()

        if cycle_result and cycle_result["success"]:
            total_saved_loop += cycle_result["details"]["detections_saved"]
            console.print(f"[green]✓ Cycle {cycle_num} successful. Saved {cycle_result['details']['detections_saved']} detection(s).[/]")
        elif cycle_result:
             console.print(f"[red]✗ Cycle {cycle_num} failed: {cycle_result['error']}[/]")
        else:
             console.print(f"[red]✗ Cycle {cycle_num} failed critically (returned None).[/]")

        if cycle_num < num_captures:
            logger.info(f"Pausing for {capture_delay} seconds."); console.print(f"\n[dim]Pause {capture_delay}s...[/]\n"); time.sleep(capture_delay)

    console.rule("[bold green]End of Autonomous Loop V6.1[/]")
    logger.info("="*30); logger.info("Metapi7_v6.1 main loop finished."); logger.info(f"Total detections saved in loop: {total_saved_loop}"); logger.info("="*30)
    console.print(f"[magenta]✓[/] Loop Session V6.1 finished. Total [bold]{total_saved_loop}[/] detections saved to '{db.DATABASE_NAME}'.")
    console.print("\n[bold green]Exiting main_loop.[/]")


# --- Point d'Entrée --- (PAS DE CHANGEMENTS ICI)
if __name__ == "__main__":
    try:
        main_loop(num_captures=2, capture_delay=3)
    except Exception as main_err:
         logger.critical(f"Unhandled exception in main_loop execution: {main_err}", exc_info=True)
         console.print(f"[bold red]UNHANDLED ERROR in main_loop:[/bold red] {main_err}")
