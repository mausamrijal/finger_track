import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load the button image
image = cv2.imread("c:/xampp/htdocs/laravelcms/hands/buttons.png")
image_height, image_width, _ = image.shape

# Define button coordinates: {button_number: (x1, y1, x2, y2)}
button_coordinates = {
    "1": (50, 50, 150, 150),
    "2": (200, 50, 300, 150),
    "3": (350, 50, 450, 150),
    "4": (50, 200, 150, 300),
    "5": (200, 200, 300, 300),
    "6": (350, 200, 450, 300),
    "7": (50, 350, 150, 450),
    "8": (200, 350, 300, 450),
    "9": (350, 350, 450, 450),
    "0": (200, 500, 300, 600),
}

# Helper function to check if a finger is on a button
def is_on_button(finger_x, finger_y, button_coords):
    x1, y1, x2, y2 = button_coords
    return x1 <= finger_x <= x2 and y1 <= finger_y <= y2

# Open video feed
cap = cv2.VideoCapture(0)

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

        # Resize the button image to match the video frame size
        image_resized = cv2.resize(image, (frame.shape[1], frame.shape[0]))

        # Overlay the resized button image on the video frame
        overlay = frame.copy()
        overlay[0:image_resized.shape[0], 0:image_resized.shape[1]] = image_resized
        frame = overlay

        # Check if hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get the tip of the index finger (landmark 8)
                index_finger = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                finger_x, finger_y = int(index_finger.x * w), int(index_finger.y * h)

                # Draw the finger position on the frame
                cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 0), -1)

                # Check if the finger is pressing a button
                for button, coords in button_coordinates.items():
                    if is_on_button(finger_x, finger_y, coords):
                        cv2.putText(frame, f"Pressed: {button}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print(f"Pressed: {button}")

        # Display the frame
        cv2.imshow('Button Press Detection', frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
