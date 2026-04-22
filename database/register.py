import face_recognition
import pickle
import os

ENCODINGS_FILE = "database/known_faces.pkl"

def register_person(name: str, image_path: str):
    """
    Encodes a person's face from a photo and saves it to the database.
    name       → The label for this person (e.g. "Alice")
    image_path → Path to a clear, front-facing photo of the person
    """

    # Load the image from disk into a NumPy array (RGB format)
    image = face_recognition.load_image_file(image_path)

    # Detect face locations in the image (returns list of bounding boxes)
    locations = face_recognition.face_locations(image)

    if len(locations) == 0:
        print(f"No face found in {image_path}")
        return

    # Extract 128-dimension face encoding (a unique numerical fingerprint)
    encodings = face_recognition.face_encodings(image, locations)
    encoding = encodings[0]  # Take the first face found

    # Load existing database or start fresh
    db_exists = os.path.exists(ENCODINGS_FILE)
    if db_exists:
        with open(ENCODINGS_FILE, "rb") as f:
            try:
                database = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                database = {"names": [], "encodings": []}
    else:
        database = {"names": [], "encodings": []}

    # Append this person's data
    database["names"].append(name)
    database["encodings"].append(encoding)

    # Save back to disk
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(database, f)

    print(f"Registered {name} successfully!")