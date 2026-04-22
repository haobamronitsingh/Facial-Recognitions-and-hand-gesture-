import face_recognition
import numpy as np

def encode_faces(frame_rgb, locations):
    """
    Converts detected face regions into 128-dimension numerical encodings.
    Each encoding is a unique "fingerprint" for that face.

    Parameters:
        frame_rgb : Image in RGB format (NumPy array)
        locations : List of (top, right, bottom, left) face bounding boxes
                    — must come from detector.py detect_faces()

    Returns:
        List of NumPy arrays, one per face.
        Each array has exactly 128 float values.
        Returns [] if no faces or locations is empty.

    Example:
        [array([0.12, -0.08, 0.34, ...]), ...]
    """
    if len(locations) == 0:
        return []

    encodings = face_recognition.face_encodings(frame_rgb, locations)
    return encodings


def encode_single_face(image_path):
    """
    Loads an image from disk and encodes the first face found.
    Used during registration — for adding new people to the database.

    Parameters:
        image_path : Full path to a JPEG or PNG photo

    Returns:
        NumPy array of 128 floats if a face is found
        None if no face is detected in the image

    Example:
        encoding = encode_single_face("photos/alice.jpg")
    """
    image = face_recognition.load_image_file("D:\facial_recog\image\IMG-20240924-WA0000.jpg")
    locations = face_recognition.face_locations(image)

    if len(locations) == 0:
        print(f"No face found in: {"D:\facial_recog\image\IMG-20240924-WA0000.jpg"}")
        return None

    # Take only the first face found in the photo
    encoding = face_recognition.face_encodings(image, locations)[0]
    return encoding


def encodings_are_valid(encodings):
    """
    Validates that a list of encodings is not empty and
    each encoding has exactly 128 dimensions.

    Parameters:
        encodings : List of NumPy arrays

    Returns:
        True if all encodings are valid 128-dimension vectors
        False if list is empty or any encoding has wrong shape
    """
    if len(encodings) == 0:
        return False
    for enc in encodings:
        if enc.shape != (128,):
            return False
    return True