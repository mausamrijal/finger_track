import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open video feed
cap = cv2.VideoCapture(0)

def fingers_up(hand_landmarks):
    """Check which fingers are up based on hand landmarks."""
    fingers = []

    # Thumb: Compare tip (4) and knuckle (2)
    if hand_landmarks[4].x > hand_landmarks[3].x:  # Right hand thumb
        fingers.append(1)
    else:
        fingers.append(0)

    # Remaining fingers: Compare tip (y) to middle knuckle (y)
    finger_tips = [8, 12, 16, 20]
    for tip in finger_tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirrored effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hands
        result = hands.process(rgb_frame)

        # Check if hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get finger states
                finger_states = fingers_up(hand_landmarks.landmark)

                # Count fingers up
                fingers_up_count = sum(finger_states)

                # Display finger info
                cv2.putText(frame, f"Fingers Up: {fingers_up_count}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Display the frame
        cv2.imshow('Hand Tracking', frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
