# list_cameras.py
import cv2
import os

def find_available_cameras(max_cameras_to_check=10):
    """
    Tente d'ouvrir des caméras de 0 à max_cameras_to_check-1.
    Affiche celles qui peuvent être ouvertes.
    """
    available_cameras = []
    print("Recherche des caméras disponibles...")
    print("------------------------------------")
    # Lister les périphériques /dev/video*
    video_devices = sorted([dev for dev in os.listdir('/dev') if dev.startswith('video')])
    if video_devices:
        print(f"Périphériques /dev/video* trouvés: {', '.join(video_devices)}")
    else:
        print("Aucun périphérique /dev/video* trouvé.")
    print("------------------------------------")
    print("Test des index de caméra avec OpenCV:")

    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  Caméra à l'index {i}: OUVERTE et frame lue (Dimensions: {frame.shape[1]}x{frame.shape[0]})")
                available_cameras.append(i)
            else:
                print(f"  Caméra à l'index {i}: OUVERTE mais impossible de lire une frame.")
            cap.release()
        # else:
        #     print(f"  Caméra à l'index {i}: Non ouverte.")
    
    if not available_cameras:
        print("\nAucune caméra n'a pu être ouverte avec succès via OpenCV dans les index testés.")
    else:
        print(f"\nCaméras OpenCV fonctionnelles trouvées aux index: {available_cameras}")
        print("L'une d'elles pourrait être votre caméra RGB, l'autre votre caméra thermique (si elle est UVC).")
        print("Notez la résolution pour vous aider à les identifier.")
    print("------------------------------------")


if __name__ == "__main__":
    find_available_cameras()
    print("\nSi votre caméra RGB (Pi Camera) et votre caméra thermique (TC001) sont connectées,")
    print("elles devraient apparaître dans la liste ci-dessus si elles sont compatibles UVC.")
    print("Utilisez l'index correspondant à la TC001 pour THERMAL_CAMERA_INDEX dans votre script principal.")