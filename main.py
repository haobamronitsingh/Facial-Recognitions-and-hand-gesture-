import cv2
import face_recognition
from camera.capture import CameraCapture
from utils.matcher import load_database, identify_face
from utils.hand_gesture import HandGestureRecognizer
from utils.media_control import MediaController
from ui.display import draw_results

ADMIN_NAME = "Ronit"
GESTURE_HOLD_FRAMES = 4


def admin_has_gesture_privilege(recognized_names):
    """Allow gesture control only when the assigned admin is recognized."""
    return ADMIN_NAME in recognized_names


def main():
    # Initialize camera (0=webcam, or "rtsp://IP:PORT/stream" for security cam)
    camera = CameraCapture(source=1)
    database = load_database()

    print("Starting facial recognition. Press 'q' to quit.")

    # Process every Nth frame to improve speed (skip frames)
    PROCESS_EVERY_N_FRAMES = 2
    frame_count = 0
    hand_recognizer = HandGestureRecognizer()
    media_controller = MediaController(cooldown_ms=1000)
    last_gesture = "none"
    stable_gesture_count = 0
    last_action = "None"

    while True:
        frame = camera.read_frame()
        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue  # Skip this frame for processing (still display it)

        # Resize for faster processing (scale down to 25% of original size)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # face_recognition uses RGB but OpenCV uses BGR — must convert!
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations in this frame
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Get encodings for all detected faces
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        recognized_names = []
        for face_encoding in face_encodings:
            name, confidence = identify_face(face_encoding, database)
            recognized_names.append(name)
            face_names.append(f"{name} ({1 - confidence:.0%})")

        gesture_privilege_enabled = admin_has_gesture_privilege(recognized_names)

        # IMPORTANT:
        # Trigger any gesture action only when this flag is True.
        # Example:
        # if gesture_detected and gesture_privilege_enabled:
        #     execute_media_command(...)

        # Scale face locations back to original frame size (we scaled by 0.25)
        scaled_locations = [
            (top * 4, right * 4, bottom * 4, left * 4)
            for (top, right, bottom, left) in face_locations
        ]

        # Draw results on the original full-size frame
        privilege_text = "ENABLED" if gesture_privilege_enabled else "LOCKED"

        gesture_name, frame = hand_recognizer.detect(frame, draw_landmarks=True)
        if gesture_name == last_gesture and gesture_name not in ("none", "unavailable"):
            stable_gesture_count += 1
        else:
            stable_gesture_count = 1
            last_gesture = gesture_name

        if gesture_privilege_enabled and stable_gesture_count >= GESTURE_HOLD_FRAMES:
            action = media_controller.trigger_for_gesture(gesture_name)
            if action:
                last_action = action

        status_lines = [
            f"Gesture Privilege: {privilege_text}",
            f"Assigned Admin: {ADMIN_NAME}",
            f"Gesture: {gesture_name}",
            f"Last Action: {last_action}",
        ]
        output_frame = draw_results(frame, scaled_locations, face_names, status_lines=status_lines)

        # Show the result in a window
        cv2.imshow("Facial Recognition System", output_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()

if __name__ == "__main__":
    main()