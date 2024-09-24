import cv2
import mediapipe as mp
import math

class poseDetector:

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity = 0  # Commencer avec la complexité minimale

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.model_complexity,
                                     smooth_landmarks=self.smooth,
                                     enable_segmentation=self.upBody,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

        self.lmList = []
        self.no_detection_counter = 0  # Compteur d'absence de détection
        self.max_attempts = 5  # Limite du nombre de tentatives de complexité

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                
                # Affichage des coordonnées avec un commentaire
                print(f"Landmark {id}: Coordonnées (X: {cx}, Y: {cy})")
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        if p1 < len(self.lmList) and p2 < len(self.lmList) and p3 < len(self.lmList):
            # Récupérer les coordonnées des points de repère
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            x3, y3 = self.lmList[p3][1:]

            # Calculer l'angle
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
            if angle < 0:
                angle += 360
            
            # Affichage de l'angle calculé avec des commentaires
            print(f"Angle calculé entre les points {p1}, {p2}, et {p3}: {angle:.2f} degrés")

            # Dessiner les points et les lignes sur l'image
            if draw:
                cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 5, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.line(img, (x3, y3), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            return angle
        else:
            print(f"Points {p1}, {p2}, {p3} non disponibles pour calculer l'angle")
            return None

    # Fonction pour ajuster la complexité du modèle et relancer la détection
    def adjustModelComplexity(self):
        if self.model_complexity < 2:
            self.model_complexity += 1
            print(f"Augmentation de la complexité du modèle à {self.model_complexity}")
        else:
            print("Complexité maximale atteinte, réinitialisation à 0")
            self.model_complexity = 0  # Réinitialiser à 0 si on atteint la complexité maximale

        # Recréer l'objet Pose avec la nouvelle complexité
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.model_complexity,
                                     smooth_landmarks=self.smooth,
                                     enable_segmentation=self.upBody,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    # Vérifier et ajuster la complexité en cas d'absence de détection
    def updateComplexityOnNoDetection(self):
        if not self.isPersonDetected():
            self.no_detection_counter += 1
            print(f"Aucune détection pendant {self.no_detection_counter} frames.")
            
            if self.no_detection_counter >= 3 and self.no_detection_counter <= self.max_attempts:
                self.adjustModelComplexity()
                self.no_detection_counter = 0  # Réinitialiser après l'ajustement de la complexité
        else:
            self.no_detection_counter = 0  # Réinitialiser si une personne est détectée

    # Fonction pour vérifier la détection d'une personne
    def isPersonDetected(self):
        return len(self.lmList) > 0

    # Fonction ajoutée pour afficher les angles du corps
    def displayBodyAngles(self, img):
        landmarks = {
            "left_elbow": [11, 13, 15],
            "right_elbow": [12, 14, 16],
        }

        for angle_name, points in landmarks.items():
            angle = self.findAngle(img, points[0], points[1], points[2])
            if angle is not None:
                cv2.putText(img, f"{angle_name}: {int(angle)}°", (points[1] - 10, points[2] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)