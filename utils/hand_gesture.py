import cv2
import json
import os
import time
import urllib.request

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - runtime dependency guard
    mp = None

DEBUG_LOG_FILE = "debug-3c6f87.log"
DEBUG_SESSION_ID = "3c6f87"


def _debug_log(run_id, hypothesis_id, location, message, data):
    # region agent log
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    # endregion


class HandGestureRecognizer:
    """Detects a single-hand gesture from a BGR frame."""

    def __init__(self, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5):
        # region agent log
        _debug_log(
            "pre-fix",
            "H1",
            "utils/hand_gesture.py:HandGestureRecognizer.__init__:entry",
            "HandGestureRecognizer initialization started",
            {
                "mpIsNone": mp is None,
                "mpModuleType": str(type(mp)),
                "mpModuleFile": getattr(mp, "__file__", None) if mp is not None else None,
            },
        )
        # endregion
        self.available = mp is not None
        self.backend = "none"
        if not self.available:
            self.hands = None
            self.drawer = None
            self.task_landmarker = None
            # region agent log
            _debug_log(
                "pre-fix",
                "H4",
                "utils/hand_gesture.py:HandGestureRecognizer.__init__:mediapipe_missing",
                "Mediapipe import unavailable",
                {"available": self.available},
            )
            # endregion
            return

        self.hands = None
        self.drawer = None
        self.task_landmarker = None

        if hasattr(mp, "solutions"):
            self.backend = "solutions"
        else:
            # Tasks-only mediapipe builds (common on newer Python versions).
            self.backend = "tasks"

        # region agent log
        _debug_log(
            "pre-fix",
            "H2",
            "utils/hand_gesture.py:HandGestureRecognizer.__init__:mp_attrs",
            "Captured mediapipe attribute availability",
            {
                "hasSolutions": hasattr(mp, "solutions"),
                "hasTasks": hasattr(mp, "tasks"),
                "backend": self.backend,
                "moduleDirSample": sorted(dir(mp))[:20],
            },
        )
        # endregion
        if self.backend == "solutions":
            self._mp_hands = mp.solutions.hands
            # region agent log
            _debug_log(
                "pre-fix",
                "H3",
                "utils/hand_gesture.py:HandGestureRecognizer.__init__:mp_solutions_selected",
                "mediapipe.solutions.hands resolved",
                {"mpHandsType": str(type(self._mp_hands))},
            )
            # endregion
            self.hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.drawer = mp.solutions.drawing_utils
        else:
            self._init_tasks_backend(max_num_hands, min_detection_confidence)

    def _init_tasks_backend(self, max_num_hands, min_detection_confidence):
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except Exception:
            self.available = False
            return

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        if not os.path.exists(model_path):
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(model_url, model_path)
            except Exception:
                self.available = False
                return

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.task_landmarker = vision.HandLandmarker.create_from_options(options)

    @staticmethod
    def _finger_up(landmarks, tip_idx, pip_idx):
        # Smaller y means visually higher on the frame for MediaPipe landmarks.
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    @staticmethod
    def _thumb_up(landmarks):
        # Thumb up: thumb tip above wrist and thumb is relatively extended.
        wrist_y = landmarks[0].y
        thumb_tip_y = landmarks[4].y
        thumb_ip_y = landmarks[3].y
        return thumb_tip_y < thumb_ip_y < wrist_y

    @staticmethod
    def _thumb_down(landmarks):
        wrist_y = landmarks[0].y
        thumb_tip_y = landmarks[4].y
        thumb_ip_y = landmarks[3].y
        return thumb_tip_y > thumb_ip_y > wrist_y

    def _classify(self, hand_landmarks):
        lms = hand_landmarks.landmark

        index_up = self._finger_up(lms, 8, 6)
        middle_up = self._finger_up(lms, 12, 10)
        ring_up = self._finger_up(lms, 16, 14)
        pinky_up = self._finger_up(lms, 20, 18)
        extended_count = sum([index_up, middle_up, ring_up, pinky_up])

        # Priority order minimizes confusion across similar poses.
        if self._thumb_up(lms) and extended_count == 0:
            return "thumb_up"
        if self._thumb_down(lms) and extended_count == 0:
            return "thumb_down"
        if extended_count == 0:
            return "fist"
        if extended_count == 4:
            return "open_palm"
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "index_up"
        if index_up and middle_up and not ring_up and not pinky_up:
            return "two_fingers"
        return "none"

    def detect(self, frame_bgr, draw_landmarks=True):
        """
        Returns: (gesture_name, frame_with_optional_landmarks)
        gesture_name: one of fist, open_palm, index_up, two_fingers, thumb_up,
                      thumb_down, none, unavailable
        """
        if not self.available:
            return "unavailable", frame_bgr

        if self.backend == "tasks":
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.task_landmarker.detect(mp_image)
            if not result.hand_landmarks:
                return "none", frame_bgr
            first_hand = result.hand_landmarks[0]
            return self._classify_from_list(first_hand), frame_bgr

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return "none", frame_bgr

        hand_landmarks = results.multi_hand_landmarks[0]
        if draw_landmarks:
            self.drawer.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                self._mp_hands.HAND_CONNECTIONS,
            )
        return self._classify(hand_landmarks), frame_bgr

    def _classify_from_list(self, landmarks):
        index_up = self._finger_up(landmarks, 8, 6)
        middle_up = self._finger_up(landmarks, 12, 10)
        ring_up = self._finger_up(landmarks, 16, 14)
        pinky_up = self._finger_up(landmarks, 20, 18)
        extended_count = sum([index_up, middle_up, ring_up, pinky_up])

        if self._thumb_up(landmarks) and extended_count == 0:
            return "thumb_up"
        if self._thumb_down(landmarks) and extended_count == 0:
            return "thumb_down"
        if extended_count == 0:
            return "fist"
        if extended_count == 4:
            return "open_palm"
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "index_up"
        if index_up and middle_up and not ring_up and not pinky_up:
            return "two_fingers"
        return "none"

