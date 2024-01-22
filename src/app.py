from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        max_num_hands=4,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            finger_count = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_label = results.multi_handedness[results.multi_hand_landmarks.index(hand_landmarks)].classification[0].label

                    hand_landmarks_list = [[landmark.x, landmark.y] for landmark in hand_landmarks.landmark]

                    if (hand_label == "Left" and hand_landmarks_list[4][0] > hand_landmarks_list[3][0]) or \
                       (hand_label == "Right" and hand_landmarks_list[4][0] < hand_landmarks_list[3][0]):
                        finger_count += 1

                    if hand_landmarks_list[8][1] < hand_landmarks_list[6][1]:  # Index finger
                        finger_count += 1
                    if hand_landmarks_list[12][1] < hand_landmarks_list[10][1]:  # Middle finger
                        finger_count += 1
                    if hand_landmarks_list[16][1] < hand_landmarks_list[14][1]:  # Ring finger
                        finger_count += 1
                    if hand_landmarks_list[20][1] < hand_landmarks_list[18][1]:  # Pinky
                        finger_count += 1

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,)

            image = cv2.flip(image, 1)

            cv2.putText(image, "Fingers: " + str(finger_count), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
