# modules/HandDetector.py
import cv2
import mediapipe as mp

class HandDetector:
    """
    Detect hands and return per-hand landmark lists and handedness.
    """

    def __init__(self, detectionCon: float = 0.7, maxHands: int = 2, trackingCon: float = 0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        # landmark ids for finger tips
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        """
        Process the (BGR) image and store results in self.results.
        We return the processed image (optionally with drawn landmarks).
        NOTE: If you want mirrored selfie preview, flip the image before calling this function.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPositions(self, img):
        """
        Return a list of hands. Each element is a dict:
        {
            "lm": [(id, x, y), ...],   # landmark pixel coords for that hand
            "handType": "Left"|"Right" or None
        }
        The order of the returned list matches MediaPipe's multi_hand_landmarks order,
        and is aligned with results.multi_handedness.
        """
        hands_out = []

        if not hasattr(self, "results") or not self.results:
            return hands_out

        if not self.results.multi_hand_landmarks:
            return hands_out

        # iterate over detected hands and pair with handedness
        for idx, handLms in enumerate(self.results.multi_hand_landmarks):
            lmList = []
            h, w, _ = img.shape
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            # try to get handedness label for the same index
            handType = None
            try:
                if self.results.multi_handedness and len(self.results.multi_handedness) > idx:
                    handType = self.results.multi_handedness[idx].classification[0].label
            except Exception:
                handType = None

            hands_out.append({"lm": lmList, "handType": handType})

        return hands_out

    def fingersUp(self, lmList, handType="Right"):
       
        if not lmList or len(lmList) < 21:
            return [0, 0, 0, 0, 0]

        lm_by_id = {t[0]: t for t in lmList}
        tips = self.tipIds
        fingers = []

        # Mirror-safe thumb logic
        thumb_tip_x = lm_by_id[tips[0]][1]
        thumb_ip_x = lm_by_id[tips[0] - 1][1]
        wrist_x = lm_by_id[0][1]

        # Instead of relying on handType label (which gets mirrored),
        # determine thumb direction relative to wrist dynamically.
        # This makes left/right work correctly even with flipped video.
        if thumb_tip_x < wrist_x:
            fingers.append(1 if thumb_tip_x < thumb_ip_x else 0)
        else:
            fingers.append(1 if thumb_tip_x > thumb_ip_x else 0)

        # Other 4 fingers
        for i in range(1, 5):
            tip_y = lm_by_id[tips[i]][2]
            pip_y = lm_by_id[tips[i] - 2][2]
            fingers.append(1 if tip_y < pip_y else 0)

        return fingers
