import cv2
from PoseModule import poseDetector  # Import de la classe poseDetector
import numpy as np

# Initialiser le détecteur de pose avec la classe poseDetector
detector = poseDetector(upBody=True)

# Fonction principale pour détecter les poses et actions techniques
def detect_combined_actions(image_path):
    print("---- Détection de la personne avec complexité ----")
    
    image = cv2.imread(image_path)

    # Utilisation de poseDetector pour tester différentes complexités
    person_detected, image_with_person = detector.tryDifferentComplexities(image)
    
    if not person_detected:
        print("Aucune personne détectée après avoir testé toutes les complexités.")
        cv2.putText(image, "Aucune personne detectee", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Resultat", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print("\n---- Détection des Actions Techniques ----")
    actions_from_movement, contours_movement, image_movement = detector.detect_actions_from_movement(image)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    actions_from_bg_subtraction, contours_bg, image_bg = detector.detect_actions_with_bg_subtraction(image, bg_subtractor)

    combined_contours = detector.remove_duplicate_actions(contours_movement, contours_bg)
    total_actions = len(combined_contours)
    print(f"\nTotal d'actions techniques détectées après fusion : {total_actions}")

    if total_actions == 0:
        print("Il ne se passe rien.")
        cv2.putText(image_with_person, "Il ne se passe rien", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image_with_person, f"Actions techniques detectees : {total_actions}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Afficher l'image annotée avec les informations
    cv2.putText(image_with_person, "Personne detectee", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Resultat", image_with_person)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Appliquer la détection sur une image
image_path = '/home/pc-camera/Bureau/Cameras/03_Code_MiniPC/images/15.jpg'  # Remplace par le chemin correct de l'image
detect_combined_actions(image_path)
