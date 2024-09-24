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
        self.model_complexity = 0  # Complexité initiale
        self.max_complexity = 2  # Complexité maximale
        self.min_complexity = 0  # Complexité minimale
        self.pose = None
        self.updatePoseModel()  # Initialisation du modèle avec complexité initiale

        try:
            # Détection de visage (face detection)
            self.mpFace = mp.solutions.face_detection
            self.face_detection = self.mpFace.FaceDetection(min_detection_confidence=self.detectionCon)

            # Détection des mains (hand detection)
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                            max_num_hands=2,
                                            min_detection_confidence=self.detectionCon,
                                            min_tracking_confidence=self.trackCon)

        except Exception as e:
            print(f"Erreur lors de l'initialisation de Mediapipe: {e}")
            raise

        self.lmList = []

    def updatePoseModel(self):
        """Créer un nouveau modèle de pose avec la complexité actuelle"""
        print(f"Création d'un nouveau modèle avec complexité: {self.model_complexity}")
        self.pose = mp.solutions.pose.Pose(static_image_mode=self.mode,
                                           model_complexity=self.model_complexity,
                                           smooth_landmarks=self.smooth,
                                           enable_segmentation=self.upBody,
                                           min_detection_confidence=self.detectionCon,
                                           min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                mp.solutions.drawing_utils.draw_landmarks(img, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def tryDifferentComplexities(self, img):
        """Essaie différentes valeurs de complexité jusqu'à trouver la meilleure"""
        for complexity in range(self.min_complexity, self.max_complexity + 1):
            self.model_complexity = complexity
            self.updatePoseModel()  # Recréer le modèle avec la nouvelle complexité

            # Appliquer la détection de pose avec la complexité actuelle
            img = self.findPose(img, draw=False)
            self.findPosition(img)

            # Si des landmarks sont détectés, alors la personne est trouvée
            if self.lmList:
                print(f"Personne détectée avec une complexité de {self.model_complexity}")
                return True, img

        # Si aucune détection après avoir testé toutes les complexités
        print("Aucune personne détectée après avoir testé toutes les complexités.")
        return False, img

    def faceDetector(self, img):
        """Détecte les visages dans l'image, affiche un carré et la probabilité"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(imgRGB)

        # Si un visage est détecté, dessiner un carré autour et afficher la probabilité
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                # Dessiner un carré autour du visage
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                # Afficher la probabilité de détection du visage
                confidence = detection.score[0]
                cv2.putText(img, f"{int(confidence * 100)}% Face", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            return True, img  # Continuer le traitement si un visage est détecté
        return False, img

    def detectGrasping(self, img):
        """Détecte si une main saisit un objet (grasping)"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                # Extraction des positions des articulations des doigts
                hand_points = [(lm.x, lm.y) for lm in handLms.landmark]
                
                # Vérification de la position des doigts
                if self.isGrasping(hand_points):
                    cv2.putText(img, "Grasping Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    return True, img
        return False, img

    def isGrasping(self, hand_points):
        """Logique simplifiée pour détecter si la main est en train de saisir (pli des doigts)"""
        # Vérification des conditions pour le grasping
        # L'index 8 correspond à l'extrémité de l'index, et l'index 6 au milieu de l'index
        # Si l'extrémité du doigt est plus proche de la paume que le milieu du doigt -> doigt plié
        index_flexion = hand_points[8][1] > hand_points[6][1]  # L'index est plié

        # Même logique pour les autres doigts
        middle_flexion = hand_points[12][1] > hand_points[10][1]  # Le majeur est plié
        ring_flexion = hand_points[16][1] > hand_points[14][1]  # L'annulaire est plié
        pinky_flexion = hand_points[20][1] > hand_points[18][1]  # L'auriculaire est plié

        # Si la majorité des doigts sont pliés, on considère cela comme une prise en main (grasping)
        return index_flexion and middle_flexion and ring_flexion and pinky_flexion

    def displayBodyAngles(self, img):
        """Affiche les angles des articulations du corps (ex. coudes, genoux)"""
        landmarks = {
            "left_elbow": [11, 13, 15],  # Épaule gauche, coude gauche, poignet gauche
            "right_elbow": [12, 14, 16],  # Épaule droite, coude droit, poignet droit
        }

        for angle_name, points in landmarks.items():
            angle = self.findAngle(img, points[0], points[1], points[2])
            if angle is not None:
                cv2.putText(img, f"{angle_name}: {int(angle)}°", (points[1] - 10, points[2] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

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

            if draw:
                cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 5, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.line(img, (x3, y3), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            return angle
        else:
            return None
