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
        self.lmList = []

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
        """Applique la détection de pose"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                mp.solutions.drawing_utils.draw_landmarks(img, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """Récupérer la position des landmarks détectés dans l'image"""
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z  # Coordonnées 3D (x, y, z)
                self.lmList.append([id, cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def tryDifferentComplexities(self, img):
        """Essaie différentes valeurs de complexité jusqu'à trouver la meilleure"""
        for complexity in range(self.min_complexity, self.max_complexity + 1):
            self.model_complexity = complexity
            self.updatePoseModel()  # Recréer le modèle avec la nouvelle complexité

            img = self.findPose(img, draw=False)
            self.findPosition(img)
            if self.lmList:
                print(f"Personne détectée avec une complexité de {self.model_complexity}")
                return True, img
        print("Aucune personne détectée après avoir testé toutes les complexités.")
        return False, img

    # Nouvelle fonction pour détecter les actions techniques par contours
    def detect_actions_from_movement(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
        _, thresh_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_contours = []
        nb_actions = 0
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > 1000:
                nb_actions += 1
                detected_contours.append(contour)
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        print(f"Nombre d'actions techniques détectées (Contours) : {nb_actions}")
        return nb_actions, detected_contours, image

    # Nouvelle fonction pour la soustraction d'arrière-plan
    def detect_actions_with_bg_subtraction(self, image, bg_subtractor):
        fg_mask = bg_subtractor.apply(image)
        _, thresh_image = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_contours = []
        nb_actions = 0
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                nb_actions += 1
                detected_contours.append(contour)
                cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)

        print(f"Nombre d'actions techniques détectées (Soustraction d'Arrière-Plan) : {nb_actions}")
        return nb_actions, detected_contours, image

    # Nouvelle fonction pour éviter les doublons
    def remove_duplicate_actions(self, contours1, contours2):
        final_contours = contours1.copy()
        for contour2 in contours2:
            duplicate = False
            for contour1 in contours1:
                distance = cv2.pointPolygonTest(contour1, tuple(contour2[0][0]), True)
                if abs(distance) < 50:
                    duplicate = True
                    break
            if not duplicate:
                final_contours.append(contour2)
        return final_contours
