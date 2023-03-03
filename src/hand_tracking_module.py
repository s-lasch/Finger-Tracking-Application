import cv2
import mediapipe as mp


class HandDetection():
    def __init__(self, mode=False, max_hands=2, complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.complexity, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    
    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, landmarks, self.mpHands.HAND_CONNECTIONS)

        return img  


    def get_fingers(self):
        finger_count = 0
        positions = []

        if self.results.multi_hand_landmarks:

            results = self.results.multi_hand_landmarks

            for hand_landmarks in results:
                hand_index = results.index(hand_landmarks)
                hand_label = self.results.multi_handedness[hand_index].classification[0].label

                for landmarks in hand_landmarks.landmark:
                    positions.append([landmarks.x, landmarks.y])

                # thumb
                if hand_label == "Left" and positions[4][0] > positions[2][0]:
                    finger_count += 1
                elif hand_label == "Right" and positions[4][0] < positions[2][0]:
                    finger_count += 1
                    
                # pinkie finger
                if positions[20][1] < positions[18][1]:
                    finger_count += 1
                
                # ring finger
                if positions[16][1] < positions[14][1]:
                    finger_count += 1

                # middle finger
                if positions[12][1] < positions[10][1]:
                    finger_count += 1

                # index finger
                if positions[8][1] < positions[6][1]:
                    finger_count += 1

        return finger_count
