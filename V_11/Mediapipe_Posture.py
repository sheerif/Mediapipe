import PoseModule as pm
import cv2

# Chargement de l'image
jpg = '/home/pc-camera/Bureau/Cameras/03_Code_MiniPC/07.jpg'
img = cv2.imread(jpg)

# Initialisation du détecteur de pose avec détection de visage/personne
detector = pm.poseDetector()

# Vérification que l'image a été correctement chargée
if img is not None:
    # Étape 1 : Détection de personne (utilise face_detection)
    detection_success = detector.detectPerson(img)
    detector.updateComplexityOnDetection(detection_success)

    # Étape 2 : Si une personne est détectée, appliquer la détection de pose
    if detection_success:
        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        # Afficher les angles du corps
        detector.displayBodyAngles(img)
    else:
        print("Aucune personne détectée.")

    # Afficher l'image avec les angles et informations
    cv2.putText(img, f'Model Complexity: {detector.model_complexity}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("Output Image with Angles", img)
    cv2.waitKey(0)
else:
    print("Erreur : l'image n'a pas été correctement chargée.")