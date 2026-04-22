import time

try:
    import pyautogui
except ImportError:  # pragma: no cover - runtime dependency guard
    pyautogui = None


class MediaController:
    """Maps gestures to media keys with per-action cooldown."""

    def __init__(self, cooldown_ms=1000):
        self.cooldown_ms = cooldown_ms
        self._last_action_ms = {}

    def _can_trigger(self, action_name):
        now_ms = int(time.time() * 1000)
        last_ms = self._last_action_ms.get(action_name, 0)
        if now_ms - last_ms < self.cooldown_ms:
            return False
        self._last_action_ms[action_name] = now_ms
        return True

    @staticmethod
    def _send_media_key(key_name):
        if pyautogui is None:
            return False
        pyautogui.press(key_name)
        return True

    def trigger_for_gesture(self, gesture_name):
        """
        Returns action label or None if no action was triggered.
        Gesture mapping:
          fist -> playpause
          thumb_up -> volumeup
          thumb_down -> volumedown
          index_up -> nexttrack
          two_fingers -> prevtrack
        """
        mapping = {
            "fist": ("Play/Pause", "playpause"),
            "thumb_up": ("Volume Up", "volumeup"),
            "thumb_down": ("Volume Down", "volumedown"),
            "index_up": ("Next Track", "nexttrack"),
            "two_fingers": ("Previous Track", "prevtrack"),
        }

        if gesture_name not in mapping:
            return None

        action_label, media_key = mapping[gesture_name]
        if not self._can_trigger(action_label):
            return None

        sent = self._send_media_key(media_key)
        if not sent:
            return "Install pyautogui for media control"
        return action_label

