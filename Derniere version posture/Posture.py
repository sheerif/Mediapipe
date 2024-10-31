import PoseModule as pm
import cv2
import time

detector = pm.poseDetector()
cap = pm.init_video_capture(0)

pTime = 0
while cap:
    success, img = cap.read()
    if not success or img is None:
        print("Failed to capture image, exiting.")
        break

    img = detector.findPose(img)
    
    # Optionally: draw and analyze the pose landmarks
    lmList = detector.findPosition(img)

    if lmList:
        print(lmList)
    
    # Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS : {int(fps)}', (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Show the frame with landmarks and angles
    cv2.imshow("Pose Detection", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
