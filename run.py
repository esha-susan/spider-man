# run.py
import cv2
from modules.HandDetector import HandDetector

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: camera not available")
        return

    detector = HandDetector(detectionCon=0.7, maxHands=2)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Mirror the frame so it feels like a selfie camera (makes handedness intuitive)
        frame = cv2.flip(frame, 1)

        # Process and draw landmarks
        display = detector.findHands(frame, draw=True)

        # Get per-hand landmark lists + handedness
        hands = detector.findPositions(display)

        # Show counts for each detected hand
        y_offset = 30
        for i, hand in enumerate(hands):
            lm = hand["lm"]
            handType = hand["handType"] or "Unknown"
            fingers = detector.fingersUp(lm, handType)
            total = fingers.count(1)

            text = f"{handType} hand #{i+1}: {total}"
            # put text with vertical offset so multiple hands stack
            cv2.putText(display, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            y_offset += 30

        # If no hands, show instruction
        if not hands:
            cv2.putText(display, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Finger Counter", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            print("👋 Exiting program...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
