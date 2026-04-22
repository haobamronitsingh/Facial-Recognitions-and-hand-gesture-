import cv2

class CameraCapture:
    def __init__(self, source=1):
        """
        source=0        → Default webcam (built-in)
        source=1        → External USB webcam
        source="rtsp://..." → IP/Security camera stream
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)  # Opens the video device

        # Set resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read_frame(self):
        """
        Reads one frame from the camera.
        ret  → True if frame was captured successfully
        frame → The actual image as a NumPy array (height x width x 3 BGR channels)
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")
        return frame

    def release(self):
        """Always release the camera when done to free resources"""
        self.cap.release()
        cv2.destroyAllWindows()