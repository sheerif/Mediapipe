import cv2
import numpy as np
import csv

# Load the image
image = cv2.imread('/home/pc-camera/Bureau/Cameras/03_Code_MiniPC/images/09.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocess (e.g., apply Gaussian blur)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Skeleton detection (e.g., OpenPose or MediaPipe integration)
# For simplicity, assume you already have the skeleton points
# Example: Skeleton points might be a list of (x, y) coordinates of key body points

skeleton_points = [(100, 200), (150, 250), (200, 300)]  # Example points (to replace with actual tracking)

# Compute angles (e.g., elbow, shoulder)
# Example calculation of the angle between three points (shoulder, elbow, wrist)
def calculate_angle(p1, p2, p3):
    angle = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    return np.abs(angle * 180.0 / np.pi)

shoulder = skeleton_points[0]
elbow = skeleton_points[1]
wrist = skeleton_points[2]

angle = calculate_angle(shoulder, elbow, wrist)

# Classify the angle into risk categories (green, orange, red)
if angle < 16:
    risk = "Zone verte"
elif 17 <= angle <= 25:
    risk = "Zone orange"
else:
    risk = "Zone rouge"

# Save results to a CSV file
with open('ergonomic_analysis.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Articulation", "Angle", "Classification"])
    writer.writerow(["Elbow", angle, risk])

print(f"L'angle de flexion du coude est de {angle}°, classé dans la {risk}")