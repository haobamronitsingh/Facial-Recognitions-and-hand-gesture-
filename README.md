### Facial recognition and hand gesture

### Summary of the Setup Guide:

* **first step :**  pip install opencv-python face_recognition numpy pillow
* **Visual Studio Build Tools 2022:** Explains how to install the "Desktop development with C++" workload, which is essential for compiling the C++ code within `dlib`.

* **Sequential Installation:** Prioritizes installing `cmake` before `dlib` to ensure the compilation process has the necessary build tools.

* **Core Libraries:** Covers the installation of `opencv-python`, `numpy`, and `pillow` for image handling.

* **Final Step:** Includes the installation of the `face_recognition` library itself.

* **dependecies:** pip install mediapipe pyautogui (for gesture control like volume up: thumbs up and volume down: thumbs down)

### Register your data first in regiter.py
* **Step 1** open the terminal
* **step 2** python -c "from database.register import register_person; register_person('username',r'imagepath')'
*     example: python -c "from database.register import register_person; register_person('username',r'D:\facial_recog\image\image.jpeg')"
* **Note** put the images in the image folder



