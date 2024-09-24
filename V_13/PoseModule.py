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
        self.model_complexity = 1  # Complexité initiale moyenne
        self.max_complexity = 2  # Complexité maximale du modèle

        try:
            # Détection de pose
            self.mpDraw = mp.solutions.drawing_utils
            self.mpPose = mp.solutions.pose
            self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                         model_complexity=self.model_complexity,
                                         smooth_landmarks=self.smooth,
                                         enable_segmentation=self.upBody,
                                         min_detection_confidence=self.detectionCon,
                                         min_tracking_confidence=self.trackCon)

            # Détection de visage à la place de object_detection
            self.mpFace = mp.solutions.face_detection
            self.face_detection = self.mpFace.FaceDetection(min_detection_confidence=self.detectionCon)

        except Exception as e:
            print(f"Erreur lors de l'initialisation de Mediapipe: {e}")
            raise

        self.lmList = []
        self.no_detection_counter = 0
        self.detection_success_counter = 0
        self.reduction_threshold = 5  # Réduction de la complexité après 5 détections réussies
        self.max_attempts = 3  # Augmentation de la complexité après 3 échecs de détection

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

    def detectPerson(self, img):
        """Utilise face_detection pour approximer la détection de personnes"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(imgRGB)

        if results.detections:
            for detection in results.detections:
                confidence = detection.score[0]
                if confidence > self.detectionCon:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = img.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                           int(bboxC.width * w), int(bboxC.height * h)
                    cv2.rectangle(img, bbox, (255, 0, 0), 2)
                    cv2.putText(img, f"Person {int(confidence * 100)}%", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    return True
        return False

    def updateComplexityOnDetection(self, detection_success):
        """Mettre à jour la complexité en fonction de la réussite ou de l'échec de la détection"""
        if detection_success:
            self.detection_success_counter += 1
            print(f"Détection réussie pendant {self.detection_success_counter} frames.")

            # Réduire la complexité si les détections réussissent plusieurs fois
            if self.detection_success_counter >= self.reduction_threshold and self.model_complexity > 0:
                self.model_complexity -= 1
                print(f"Réduction de la complexité à {self.model_complexity} pour optimiser.")
                self.detection_success_counter = 0  # Réinitialiser le compteur

        else:
            self.no_detection_counter += 1
            print(f"Échec de détection pendant {self.no_detection_counter} frames.")

            # Augmenter la complexité après plusieurs échecs
            if self.no_detection_counter >= self.max_attempts:
                if self.model_complexity < self.max_complexity:
                    self.model_complexity += 1
                    print(f"Augmentation de la complexité à {self.model_complexity}")
                else:
                    print("Complexité maximale atteinte, réinitialisation à 0")
                    self.model_complexity = 0
                self.no_detection_counter = 0  # Réinitialiser le compteur après l'ajustement
