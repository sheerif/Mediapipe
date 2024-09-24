import PoseModule as pm
import cv2

# Chargement de l'image
jpg = '/home/pc-camera/Bureau/Cameras/03_Code_MiniPC/14.jpg'
img = cv2.imread(jpg)

# Initialisation du détecteur de pose
detector = pm.poseDetector()

# Vérification que l'image a été correctement chargée
if img is not None:
    # Détection de la pose
    img = detector.findPose(img)
    lmList = detector.findPosition(img)

    # Si aucune personne n'est détectée, ajuster la complexité du modèle
    detector.updateComplexityOnNoDetection()

    # Si une personne est détectée, afficher les angles du corps
    if detector.isPersonDetected():
        print("Personne détectée !")
        detector.displayBodyAngles(img)
    else:
        print("Aucune personne détectée.")

    # Afficher l'image avec les angles et autres informations
    cv2.putText(img, f'Model Complexity: {detector.model_complexity}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("Output Image with Angles", img)
    cv2.waitKey(0)
else:
    print("Erreur : l'image n'a pas été correctement chargée.")