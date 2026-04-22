import face_recognition

def detect_faces(frame_rgb, model="hog"):
    """
    Detects face locations in an RGB image frame.

    Parameters:
        frame_rgb  : Image in RGB format (NumPy array) — NOT BGR
        model      : "hog"  → fast, works on CPU, good for real-time webcam
                     "cnn"  → accurate, needs GPU, better for security cameras

    Returns:
        List of tuples: [(top, right, bottom, left), ...]
        Each tuple is the bounding box of one detected face.

    Example return:
        [(120, 300, 200, 220)]  ← one face found
        []                      ← no faces found
    """
    locations = face_recognition.face_locations(frame_rgb, model=model)
    return locations


def is_face_detected(frame_rgb, model="hog"):
    """
    Simple boolean check — returns True if at least one face is in the frame.
    Useful for triggering alerts or skipping encoding when no face is present.

    Parameters:
        frame_rgb : Image in RGB format
        model     : "hog" or "cnn"

    Returns:
        True if one or more faces found, False otherwise
    """
    locations = detect_faces(frame_rgb, model=model)
    return len(locations) > 0


def filter_small_faces(locations, min_size=50):
    """
    Removes faces that are too small to reliably recognize.
    Small faces = person is too far from the camera.

    Parameters:
        locations : List of (top, right, bottom, left) tuples
        min_size  : Minimum face height in pixels (default 50px)

    Returns:
        Filtered list of locations where face height >= min_size
    """
    filtered = []
    for (top, right, bottom, left) in locations:
        face_height = bottom - top   # height of the bounding box in pixels
        if face_height >= min_size:
            filtered.append((top, right, bottom, left))
    return filtered