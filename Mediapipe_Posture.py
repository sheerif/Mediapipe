import PoseModule as pm
import cv2

# Chargement de l'image
<<<<<<< HEAD
jpg = '/home/pc-camera/Bureau/Cameras/03_Code_MiniPC/images/02.jpg'
img = cv2.imread(jpg)

# Initialisation du détecteur de pose avec détection de visage/personne et main (grasping)
=======
jpg = '/home/pc-camera/Bureau/Cameras/03_Code_MiniPC/07.jpg'
img = cv2.imread(jpg)

# Initialisation du détecteur de pose avec détection de visage/personne et main
>>>>>>> ea4b0d0b024278fa7b79c01727cb0ec72e8be57a
detector = pm.poseDetector()

# Vérification que l'image a été correctement chargée
if img is not None:
    # Étape 1 : Effectuer le traitement avec le modèle de complexité
    detection_success, img = detector.tryDifferentComplexities(img)

    # Si une personne est détectée, afficher les angles du corps
    if detection_success:
<<<<<<< HEAD
        if detector.lmList:  # On vérifie que des landmarks sont présents
            detector.displayBodyAngles(img)
        else:
            print("Aucun point de repère (landmarks) détecté pour afficher les angles.")
=======
        detector.displayBodyAngles(img)
>>>>>>> ea4b0d0b024278fa7b79c01727cb0ec72e8be57a

    # Étape 2 : Effectuer la détection de visage après le traitement du modèle de complexité
    face_detected, img = detector.faceDetector(img)

    # Si un visage est détecté, afficher le carré et la probabilité
    if face_detected:
        print("Visage détecté après le traitement du modèle de complexité.")
    else:
        print("Aucun visage détecté après le traitement du modèle de complexité.")

    # Étape 3 : Détection de la prise en main (grasping) après la détection de visage
    grasping_detected, img = detector.detectGrasping(img)

    if grasping_detected:
        print("Prise en main détectée.")
    else:
        print("Aucune prise en main détectée.")

    # Afficher l'image avec les angles et informations
    cv2.putText(img, f'Model Complexity: {detector.model_complexity}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.imshow("Output Image with Angles, Face, and Grasping Detection", img)
    cv2.waitKey(0)
else:
    print("Erreur : l'image n'a pas été correctement chargée.")
