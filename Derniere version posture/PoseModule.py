import cv2
import mediapipe as mp
import math

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5, complexity=1):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.complexity = complexity
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = None
        self.init_pose()

    def init_pose(self):
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth,
            enable_segmentation=self.upBody,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)
        print(f"Model complexity set to {self.complexity}")

    def findAngle(self, img, p1, p2, p3, draw=True):
        if p1 >= len(self.lmList) or p2 >= len(self.lmList) or p3 >= len(self.lmList):
            return None

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle

        # Print angle in console
        print(f"Angle between {p1}, {p2}, {p3}: {angle:.2f} degrees")

        # Draw the angle on the image
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f'{int(angle)}°', (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return angle

    def adjust_complexity_based_on_quality(self, landmarks, img):
        original_complexity = self.complexity
        image_quality = self.assess_image_quality(img)
        landmark_count = len(landmarks)

        if image_quality < 50 and landmark_count < 10 and self.complexity > 0:
            self.complexity = 0
            self.init_pose()
        elif image_quality > 70 and landmark_count > 15 and self.complexity < 2:
            self.complexity = 2
            self.init_pose()

        if self.complexity != original_complexity:
            print(f"Complexity adjusted from {original_complexity} to {self.complexity}")
            cv2.putText(img, f'Complexity: {self.complexity}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def assess_image_quality(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        return image_variance

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        landmarks = self.findPosition(img, False)

        # Ajuster la complexité du modèle en fonction de la qualité et du nombre de landmarks
        self.adjust_complexity_based_on_quality(landmarks, img)

        # Calculer et afficher les angles pour les parties du haut du corps
        if len(landmarks) > 0:
            # Angle Tête -> Épaule gauche -> Coude gauche
            self.findAngle(img, 0, 11, 13)  # Nez (0), épaule gauche (11), coude gauche (13)
            
            # Angle Tête -> Épaule droite -> Coude droit
            self.findAngle(img, 0, 12, 14)  # Nez (0), épaule droite (12), coude droit (14)
            
            # Angle Épaule gauche -> Coude gauche -> Poignet gauche
            self.findAngle(img, 11, 13, 15)  # Épaule gauche (11), coude gauche (13), poignet gauche (15)
            
            # Angle Épaule droite -> Coude droit -> Poignet droit
            self.findAngle(img, 12, 14, 16)  # Épaule droite (12), coude droit (14), poignet droit (16)
            
            # Angle Épaule gauche -> Hanche gauche -> Genou gauche
            self.findAngle(img, 11, 23, 25)  # Épaule gauche (11), hanche gauche (23), genou gauche (25)
            
            # Angle Épaule droite -> Hanche droite -> Genou droit
            self.findAngle(img, 12, 24, 26)  # Épaule droite (12), hanche droite (24), genou droit (26)

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
        self.lmList = lmList  # Store landmark list for angle calculations
        return lmList

def open_camera(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source {source}")
        return None, True
    return cap, False

def init_video_capture(primary_source, backup_source=None):
    cap, failed = open_camera(primary_source)
    if failed and backup_source:
        print("Primary source failed, trying backup source.")
        cap, failed = open_camera(backup_source)
    if failed:
        return None
    return cap
