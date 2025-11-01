import cv2
import mediapipe as mp
class HandDetector():
    def __init__(self,mode=False,max_hands=2):
        self.mode=mode
        self.max_hands=max_hands
        self.capture=cv2.VideoCapture(0)
        self.mpHands=mp.solutions.hands
        self.hands = self.mpHands.Hands(
    static_image_mode=self.mode,
    max_num_hands=self.max_hands,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

        self.mpDraw=mp.solutions.drawing_utils
    def fdHands(self,frame,draw=True):
        self.results=self.hands.process(frame)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)
                else:
                    print("Not Drawing")
        return frame
    def fdPositions(self,frame):
        landmarks_list=[]
        if self.results.multi_hand_landmarks:
            hand=self.results.multi_hand_landmarks[0]
            for id,lm in enumerate(hand.landmark):
                h,w,_=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                landmarks_list.append([id,cx,cy])
        return landmarks_list