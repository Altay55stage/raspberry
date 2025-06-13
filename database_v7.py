# database_v7.py
import sqlite3
import os
import logging
from datetime import datetime

# --- Configuration de la Base de Données ---
BASE_DIR = '/home/acevik' # Assurez-vous que c'est le même que dans votre script principal
DB_DIR = os.path.join(BASE_DIR, 'db')
DATABASE_NAME = 'metapi_detections_v7.db'
DB_PATH = os.path.join(DB_DIR, DATABASE_NAME)

logger = logging.getLogger(__name__) # Utilise le logger configuré dans le script principal si importé, sinon configure un basique.
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

def get_db_connection():
    """Établit une connexion à la base de données SQLite."""
    if not os.path.exists(DB_DIR):
        try:
            os.makedirs(DB_DIR)
            logger.info(f"Created database directory: {DB_DIR}")
        except OSError as e:
            logger.error(f"Failed to create database directory {DB_DIR}: {e}", exc_info=True)
            raise  # Propage l'erreur si le dossier ne peut être créé

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # Permet d'accéder aux colonnes par nom
    logger.debug(f"Database connection established to {DB_PATH}")
    return conn

def init_db():
    """Initialise la base de données et crée la table 'detections' si elle n'existe pas."""
    logger.info(f"Initializing database schema for {DB_PATH}...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Création de la table 'detections' avec les nouveaux champs thermiques
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_filename TEXT NOT NULL,           -- Nom de fichier de l'image RGB
                module_lat REAL,
                module_lon REAL,
                module_alt REAL,
                module_fix_time TEXT,                   -- Timestamp du fix GPS du module
                drone_heading REAL,                     -- Cap du drone en degrés
                drone_speed_knots REAL,                 -- Vitesse du drone en nœuds
                person_id_in_image INTEGER,             -- Identifiant unique de la personne dans CETTE image
                confidence REAL,                        -- Confiance de la détection YOLO
                bbox_x1 INTEGER, 
                bbox_y1 INTEGER, 
                bbox_x2 INTEGER, 
                bbox_y2 INTEGER,                        -- Coordonnées de la bounding box
                bbox_center_x_px INTEGER,
                bbox_center_y_px INTEGER,
                estimated_lat REAL,                     -- Latitude estimée de la personne
                estimated_lon REAL,                     -- Longitude estimée de la personne
                estimated_distance_m REAL,              -- Distance estimée à la personne en mètres
                map_link TEXT,                          -- Lien Google Maps vers la position estimée
                estimation_status TEXT,                 -- Statut de l'estimation (e.g., Estimated, NoGPSFix, EstimFail_NoHdg)
                
                -- Champs pour l'intégration thermique (V7)
                thermal_confirmed BOOLEAN,              -- True si la détection est confirmée thermiquement, False si non, NULL si non vérifié
                thermal_status_detail TEXT,             -- Plus de détails sur la vérification thermique (e.g., AvgTemp:35.2C, NotChecked, DataUnavailable)
                thermal_image_filename TEXT             -- Nom de fichier de l'image thermique associée (si applicable)
            )
        ''')
        logger.info("Table 'detections' checked/created successfully.")

        # Vous pourriez vouloir ajouter des index pour améliorer les performances des requêtes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections (timestamp);
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_detections_image_filename ON detections (image_filename);
        ''')
        logger.info("Indexes checked/created successfully.")

        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()
    logger.info("Database initialization complete.")

def save_detection(detection_data):
    """
    Enregistre une nouvelle détection dans la base de données.
    detection_data: Un dictionnaire contenant les données de la détection.
    Retourne l'ID de la ligne insérée, ou None en cas d'échec.
    """
    logger.debug(f"Attempting to save detection: {detection_data.get('image_filename')}, Person ID: {detection_data.get('person_id_in_image')}")
    
    # Vérification des champs obligatoires (exemple)
    if not detection_data.get('image_filename'):
        logger.error("Save detection failed: 'image_filename' is missing.")
        return None

    sql = ''' INSERT INTO detections(
                timestamp, image_filename, module_lat, module_lon, module_alt, module_fix_time,
                drone_heading, drone_speed_knots, person_id_in_image, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_center_x_px, bbox_center_y_px,
                estimated_lat, estimated_lon, estimated_distance_m, map_link, estimation_status,
                thermal_confirmed, thermal_status_detail, thermal_image_filename
              )
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) '''
    
    current_time = datetime.now()

    values = (
        detection_data.get('detection_timestamp', current_time), # Permet de passer un timestamp spécifique si besoin, sinon now()
        detection_data.get('image_filename'),
        detection_data.get('module_lat'),
        detection_data.get('module_lon'),
        detection_data.get('module_alt'),
        detection_data.get('module_fix_time'),
        detection_data.get('drone_heading'),
        detection_data.get('drone_speed_knots'),
        detection_data.get('person_id_in_image'),
        detection_data.get('confidence'),
        detection_data.get('bbox_x1'),
        detection_data.get('bbox_y1'),
        detection_data.get('bbox_x2'),
        detection_data.get('bbox_y2'),
        detection_data.get('bbox_center_x_px'),
        detection_data.get('bbox_center_y_px'),
        detection_data.get('estimated_lat'),
        detection_data.get('estimated_lon'),
        detection_data.get('estimated_distance_m'),
        detection_data.get('map_link'),
        detection_data.get('estimation_status'),
        detection_data.get('thermal_confirmed'),         # Nouveau champ V7
        detection_data.get('thermal_status_detail'),    # Nouveau champ V7
        detection_data.get('thermal_image_filename')    # Nouveau champ V7
    )

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()
        last_id = cursor.lastrowid
        logger.info(f"Detection saved successfully. Image: {detection_data.get('image_filename')}, DB ID: {last_id}, Person ID: {detection_data.get('person_id_in_image')}, Thermal: {detection_data.get('thermal_confirmed')}")
        return last_id
    except sqlite3.Error as e:
        logger.error(f"Failed to save detection for image {detection_data.get('image_filename')}: {e}", exc_info=True)
        if conn: # Tenter de rollback en cas d'erreur si la connexion est toujours active
            try:
                conn.rollback()
                logger.info("Transaction rolled back.")
            except sqlite3.Error as rb_err:
                logger.error(f"Error during rollback: {rb_err}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()

def get_all_detections(limit=None):
    """Récupère toutes les détections, avec une limite optionnelle."""
    logger.debug(f"Fetching all detections (limit: {limit})...")
    detections = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM detections ORDER BY timestamp DESC"
        if limit and isinstance(limit, int) and limit > 0:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            detections.append(dict(row)) # Convertit chaque sqlite3.Row en dictionnaire
        logger.info(f"Fetched {len(detections)} detections.")
    except sqlite3.Error as e:
        logger.error(f"Error fetching detections: {e}", exc_info=True)
    finally:
        if 'conn' in locals() and conn:
            conn.close()
    return detections

def get_detection_by_id(detection_id):
    """Récupère une détection spécifique par son ID."""
    logger.debug(f"Fetching detection with ID: {detection_id}...")
    detection = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM detections WHERE id = ?", (detection_id,))
        row = cursor.fetchone()
        if row:
            detection = dict(row)
            logger.info(f"Detection ID {detection_id} fetched successfully.")
        else:
            logger.info(f"No detection found with ID: {detection_id}")
    except sqlite3.Error as e:
        logger.error(f"Error fetching detection ID {detection_id}: {e}", exc_info=True)
    finally:
        if 'conn' in locals() and conn:
            conn.close()
    return detection

# --- Point d'entrée pour tests ou initialisation manuelle ---
if __name__ == "__main__":
    print(f"Database module v7. DB Path: {DB_PATH}")
    
    # Test d'initialisation
    try:
        init_db()
        print("Database initialized/verified successfully.")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        exit(1)

    # Test d'insertion (exemple)
    print("\nAttempting to save a test detection...")
    test_data = {
        'image_filename': 'test_rgb_image_001.jpg',
        'module_lat': 45.123456,
        'module_lon': -75.654321,
        'module_alt': 100.5,
        'module_fix_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        'drone_heading': 180.0,
        'drone_speed_knots': 5.2,
        'person_id_in_image': 1,
        'confidence': 0.85,
        'bbox_x1': 100, 'bbox_y1': 150, 'bbox_x2': 200, 'bbox_y2': 350,
        'bbox_center_x_px': 150, 'bbox_center_y_px': 250,
        'estimated_lat': 45.123500,
        'estimated_lon': -75.654300,
        'estimated_distance_m': 25.5,
        'map_link': 'http://example.com/map?lat=45.123500&lon=-75.654300',
        'estimation_status': 'Estimated',
        'thermal_confirmed': True,
        'thermal_status_detail': 'AvgTemp:36.5C',
        'thermal_image_filename': 'test_thermal_image_001.png'
    }
    
    inserted_id = save_detection(test_data)
    if inserted_id:
        print(f"Test detection saved with ID: {inserted_id}")

        # Test de récupération
        print("\nFetching the test detection...")
        retrieved_detection = get_detection_by_id(inserted_id)
        if retrieved_detection:
            print("Retrieved detection data:")
            for key, value in retrieved_detection.items():
                print(f"  {key}: {value}")
        else:
            print("Failed to retrieve test detection.")

        print("\nFetching all detections (limit 5)...")
        all_dets = get_all_detections(limit=5)
        if all_dets:
            print(f"Found {len(all_dets)} detections:")
            for i, det in enumerate(all_dets):
                print(f"  Detection {i+1}: ID={det['id']}, Img={det['image_filename']}, ThermalConf={det['thermal_confirmed']}")
        else:
            print("No detections found or error fetching.")
    else:
        print("Failed to save test detection.")

    print("\nDatabase module test finished.")