import face_recognition
import pickle
import numpy as np
import os
import json
import time

ENCODINGS_FILE = "database/known_faces.pkl"
TOLERANCE = 0.5  # Lower = stricter matching (0.4–0.6 is typical)
DEBUG_LOG_FILE = "debug-9d74b2.log"
DEBUG_SESSION_ID = "9d74b2"


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

def load_database():
    """Load all known face encodings from disk"""
    _debug_log(
        "pre-fix",
        "H1",
        "utils/matcher.py:load_database:entry",
        "Entering load_database",
        {
            "encodingsFile": ENCODINGS_FILE,
            "exists": os.path.exists(ENCODINGS_FILE),
            "sizeBytes": os.path.getsize(ENCODINGS_FILE) if os.path.exists(ENCODINGS_FILE) else None,
        },
    )
    if not os.path.exists(ENCODINGS_FILE) or os.path.getsize(ENCODINGS_FILE) == 0:
        _debug_log(
            "post-fix",
            "HF1",
            "utils/matcher.py:load_database:empty_or_missing",
            "Database file missing or empty; returning initialized database",
            {
                "exists": os.path.exists(ENCODINGS_FILE),
                "sizeBytes": os.path.getsize(ENCODINGS_FILE) if os.path.exists(ENCODINGS_FILE) else None,
            },
        )
        return {"names": [], "encodings": []}

    with open(ENCODINGS_FILE, "rb") as f:
        _debug_log(
            "post-fix",
            "H2",
            "utils/matcher.py:load_database:before_pickle",
            "About to call pickle.load",
            {"fileHandleOpened": True},
        )
        result = pickle.load(f)
        _debug_log(
            "post-fix",
            "H3",
            "utils/matcher.py:load_database:after_pickle",
            "pickle.load succeeded",
            {
                "keys": list(result.keys()) if isinstance(result, dict) else None,
                "namesCount": len(result.get("names", [])) if isinstance(result, dict) else None,
                "encodingsCount": len(result.get("encodings", [])) if isinstance(result, dict) else None,
            },
        )
        return result

def identify_face(unknown_encoding, database):
    """
    Compares an unknown face encoding against all known faces.
    Returns the matched name or "Unknown".
    """

    known_encodings = database["encodings"]
    known_names = database["names"]
    _debug_log(
        "pre-fix",
        "H4",
        "utils/matcher.py:identify_face:counts",
        "identify_face received database",
        {"knownNamesCount": len(known_names), "knownEncodingsCount": len(known_encodings)},
    )
    if len(known_encodings) == 0:
        _debug_log(
            "post-fix",
            "HF2",
            "utils/matcher.py:identify_face:empty_database",
            "No known encodings available; returning Unknown safely",
            {"returnedName": "Unknown", "returnedDistance": 1.0},
        )
        return "Unknown", 1.0

    # face_distance() returns a float per known face
    # Smaller distance = more similar faces (0.0 = perfect match)
    distances = face_recognition.face_distance(known_encodings, unknown_encoding)

    # Find the best (lowest distance) match
    best_match_index = np.argmin(distances)
    best_distance = distances[best_match_index]

    if best_distance <= TOLERANCE:
        return known_names[best_match_index], best_distance
    else:
        return "Unknown", best_distance