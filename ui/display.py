import cv2

def draw_results(frame, face_locations, face_names, status_lines=None):
    """
    Draws colored bounding boxes and name labels on the video frame.
    face_locations → List of (top, right, bottom, left) tuples
    face_names     → List of name strings matching each location
    """
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # Choose color: green for known, red for unknown
        is_unknown = name.startswith("Unknown")
        color = (0, 0, 255) if is_unknown else (0, 255, 0)

        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness=2)

        # Draw filled label background below the box
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)

        # Write the person's name inside the label
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),    # Position: inside label box
            cv2.FONT_HERSHEY_DUPLEX,   # Font style
            0.6,                        # Font scale
            (255, 255, 255),            # White text
            1                           # Thickness
        )

    if status_lines:
        y = 30
        for line in status_lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            y += 28

    return frame