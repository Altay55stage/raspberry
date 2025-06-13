Metapi Drone Detection System v7.0

Ce projet est un système avancé de détection de personnes par drone, conçu pour fonctionner sur un Raspberry Pi. Il utilise un modèle YOLOv8 pour la détection visuelle (RGB), intègre les données d'un module GPS pour obtenir la position et le cap du drone, et utilise une caméra thermique pour valider les détections. Toutes les informations sont ensuite enregistrées dans une base de données SQLite et enrichies de métadonnées EXIF.

🌟 Fonctionnalités Principales

Détection par IA : Utilise YOLOv8 pour la détection de personnes en temps réel sur les images capturées.

Géolocalisation Avancée : Estime la position GPS de la personne détectée en se basant sur :

La position GPS du drone.

Le cap (heading) du drone pour un azimut précis.

L'altitude, le pitch et les paramètres de la caméra.

Confirmation Thermique : Croise les détections RGB avec les données d'une caméra thermique pour valider la présence d'une source de chaleur, réduisant ainsi les faux positifs.

Base de Données Robuste : Sauvegarde chaque détection dans une base de données SQLite v7, incluant les coordonnées estimées, les statuts de confirmation thermique, et les liens Google Maps.

Métadonnées Complètes : Ajoute des métadonnées EXIF détaillées à chaque image, incluant les coordonnées GPS du module, les informations de détection et les paramètres du système.

Interface Console Riche : Affiche les résultats des cycles de détection dans des tableaux clairs et formatés via la bibliothèque rich.

Configuration Flexible : Les paramètres clés (seuil de confiance, activation thermique, etc.) sont gérés via un fichier config_v7.json.

📂 Structure du Projet
/home/acevik/
├── metapi_v7.py               # Script principal du système de détection (anciennement metapi10mpro.py)
├── database_v7.py             # Module de gestion de la base de données SQLite
├── list_cameras.py            # Utilitaire pour identifier les index des caméras connectées
├── config_v7.json             # Fichier de configuration des paramètres du système
├── yolov8n.pt                 # Modèle de détection YOLOv8 (doit être téléchargé)
├── captures/                  # Dossier où les images RGB annotées sont sauvegardées
├── captures_thermal/          # Dossier où les images thermiques sont sauvegardées
└── db/
    └── metapi_detections_v7.db  # Fichier de la base de données SQLite

⚙️ Prérequis Matériels

Ce projet est conçu pour un système embarqué, typiquement un drone équipé de :

Ordinateur de bord : Un Raspberry Pi 4 (ou supérieur) est recommandé.

Caméra RGB : Une Raspberry Pi Camera Module 3 (ou compatible libcamera).

Caméra Thermique : Une caméra thermique compatible UVC (USB Video Class), comme la Top-Tek TC001. Le système la traitera comme une webcam standard.

Module GPS : Un module GPS communiquant via un port série (ex: SIMCom A7670E, SIM7600, etc.) connecté via un adaptateur USB-Série et reconnu comme /dev/ttyUSBx.

🛠️ Installation et Configuration

Suivez ces étapes pour rendre le système opérationnel.

Étape 1 : Clonage du Dépôt

Clonez ce dépôt dans le répertoire de base de votre Raspberry Pi.

cd /home/acevik
git clone <URL_DE_VOTRE_DEPOT_GIT> .
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Étape 2 : Installation des Dépendances Système

Assurez-vous que les outils libcamera et pip sont installés.

sudo apt update
sudo apt install -y libcamera-apps python3-pip python3-venv
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Étape 3 : Création d'un Environnement Virtuel et Installation des Paquets Python

Il est fortement recommandé d'utiliser un environnement virtuel.

# Se placer dans le répertoire du projet
cd /home/acevik

# Créer l'environnement virtuel
python3 -m venv venv

# Activer l'environnement virtuel
source venv/bin/activate

# Créer un fichier requirements.txt
# (Vous pouvez le créer manuellement ou utiliser celui-ci)
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Créez un fichier nommé requirements.txt avec le contenu suivant :

# requirements.txt
pyserial
opencv-python-headless
numpy
ultralytics
rich
piexif
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Txt
IGNORE_WHEN_COPYING_END

Installez ensuite les paquets :

pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
Étape 4 : Téléchargement du Modèle YOLO

Le script utilise yolov8n.pt. Le framework ultralytics le téléchargera automatiquement lors de la première exécution, mais vous pouvez aussi le faire manuellement.

Étape 5 : Connexion et Identification du Matériel

GPS : Connectez votre module GPS. Vérifiez qu'il est bien reconnu sur le port série spécifié dans le script (/dev/ttyUSB2 par défaut). Vous pouvez lister les ports avec ls /dev/ttyUSB*.

Caméras : Connectez la caméra Pi et la caméra thermique USB.

Étape 6 : Identifier l'Index de la Caméra Thermique

La caméra thermique sera vue par le système comme une caméra vidéo standard. Vous devez trouver son index (0, 1, 2...). Exécutez le script utilitaire fourni :

# Assurez-vous que l'environnement virtuel est activé
source venv/bin/activate

python list_cameras.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Le script affichera les caméras disponibles et leur résolution.

La caméra Pi a une haute résolution (ex: 1920x1080).

La caméra thermique TC001 aura une résolution plus basse (ex: 256x192).

Notez l'index de votre caméra thermique.

Étape 7 : Configuration des Scripts

C'est l'étape la plus importante. Vous devez ajuster les constantes et le fichier de configuration.

1. Configuration du script principal (metapi_v7.py)

Ouvrez metapi_v7.py et modifiez ces constantes au début du fichier si nécessaire :

Constante	Description	Valeur par défaut
SERIAL_PORT	Le port série de votre module GPS.	'/dev/ttyUSB2'
THERMAL_CAMERA_INDEX	L'index de votre caméra thermique trouvé à l'étape 6.	-1 (doit être changé)
CAMERA_PARAMS	Calibration cruciale ! Ajustez ces valeurs pour votre drone.	Voir ci-dessous

Ajustez le dictionnaire CAMERA_PARAMS avec une grande précision. L'exactitude de l'estimation GPS en dépend !

'altitude': Altitude de vol (en mètres). Peut être ajustée dynamiquement si vous avez un capteur.

'camera_yaw_relative_deg': Angle de la caméra sur l'axe horizontal par rapport à l'avant du drone. 0.0 si elle pointe droit devant.

'pitch': Angle de la caméra vers le bas (en degrés). 90.0 pour une vue parfaitement à la verticale (nadir).

'focal_length_px', 'sensor_width_px', 'sensor_height_px': Paramètres intrinsèques de votre caméra.

2. Configuration du fichier config_v7.json

Ce fichier contrôle le comportement de la détection.

{
    "detection_confidence_threshold": 0.50,
    "thermal_enabled": true,
    "thermal_confirm_human": true,
    "thermal_intensity_threshold_hot": 160,
    "thermal_camera_width": 256,
    "thermal_camera_height": 192,
    "thermal_roi_analysis_size_px": 30
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END
Paramètre	Description
detection_confidence_threshold	Seuil de confiance (entre 0.0 et 1.0) pour qu'une détection YOLO soit considérée valide.
thermal_enabled	true pour activer la capture et l'analyse thermique, false pour la désactiver.
thermal_confirm_human	Si true, une détection n'est considérée comme "chaude" que si elle dépasse le seuil d'intensité.
thermal_intensity_threshold_hot	À calibrer ! Seuil d'intensité de pixel (0-255) dans la zone d'intérêt thermique pour la considérer comme "chaude". Plus la valeur est élevée, plus la source doit être intense.
thermal_camera_width / height	Résolution de votre caméra thermique. Doit correspondre à ce que list_cameras.py a rapporté.
thermal_roi_analysis_size_px	Taille (en pixels) du carré analysé sur l'image thermique.
▶️ Exécution du Système

Une fois la configuration terminée, vous pouvez lancer le système en mode autonome.

Assurez-vous que l'environnement virtuel est activé : source venv/bin/activate.

Exécutez le script principal :

python metapi_v7.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Le script va démarrer et exécuter le nombre de cycles de détection définis dans la fonction main_loop (par défaut 2 cycles avec 3 secondes de pause).

Comprendre la Sortie

Pour chaque cycle, vous verrez :

Le statut de la connexion GPS.

La confirmation de la capture des images RGB et thermique.

Un tableau récapitulatif des détections, avec :

L'ID de la personne dans l'image.

L'ID de l'entrée dans la base de données.

La confiance de la détection.

Le statut de la confirmation thermique (OK, NOK, ou N/A).

Le statut de l'estimation de la position.

Les coordonnées GPS estimées et la distance.

Un lien cliquable vers Google Maps.

Où trouver les résultats ?

Images RGB annotées : Dans le dossier /home/acevik/captures/.

Images thermiques brutes : Dans le dossier /home/acevik/captures_thermal/.

Données structurées : Dans la base de données /home/acevik/db/metapi_detections_v7.db. Vous pouvez l'explorer avec un outil comme DB Browser for SQLite.

🗃️ Schéma de la Base de Données

La table detections contient les colonnes suivantes :

Colonne	Type	Description
id	INTEGER	Clé primaire auto-incrémentée.
timestamp	DATETIME	Horodatage de l'enregistrement dans la DB.
image_filename	TEXT	Nom du fichier de l'image RGB.
module_lat, module_lon, module_alt	REAL	Coordonnées GPS du drone au moment de la capture.
module_fix_time	TEXT	Horodatage du fix GPS du module.
drone_heading	REAL	Cap du drone en degrés (0-360).
drone_speed_knots	REAL	Vitesse du drone en nœuds.
person_id_in_image	INTEGER	ID unique de la personne dans cette image (1, 2, 3...).
confidence	REAL	Confiance de la détection YOLO (0.0 - 1.0).
bbox_...	INTEGER	Coordonnées de la boîte englobante de la détection.
estimated_lat, estimated_lon	REAL	Coordonnées GPS estimées de la personne.
estimated_distance_m	REAL	Distance horizontale estimée à la personne.
map_link	TEXT	Lien Google Maps vers la position estimée.
estimation_status	TEXT	Statut du calcul (ex: Estimated, NoGPSFix, EstimFail_NoHdg).
thermal_confirmed	BOOLEAN	True si confirmée, False si infirmée, NULL si non vérifiée.
thermal_status_detail	TEXT	Détails de l'analyse (ex: AvgIntensity:180.5, NotChecked).
thermal_image_filename	TEXT	Nom du fichier de l'image thermique associée.
