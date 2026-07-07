from dataclasses import dataclass

import cv2 as cv


@dataclass
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    right_eye: 'FaceBox' = None


class FaceTracker:
    def __init__(self, cascade_path=None, eye_cascade_path=None, smoothing=0.75, min_size=(80, 80), detection_interval=2, max_missing_frames=10):
        self.cascade_path = cascade_path or 'models/HaarCascades/haarcascade_frontalface_default.xml'
        self.eye_cascade_path = eye_cascade_path or 'models/HaarCascades/haarcascade_eye.xml'
        self.cascade = None
        self.eye_cascade = None
        self.use_cascade = hasattr(cv, 'CascadeClassifier')
        if self.use_cascade:
            self.cascade = cv.CascadeClassifier(self.cascade_path)
            self.eye_cascade = cv.CascadeClassifier(self.eye_cascade_path)
            self.use_cascade = not self.cascade.empty()
        self.smoothing = float(smoothing)
        self.min_size = min_size
        self.detection_interval = max(1, int(detection_interval))
        self.max_missing_frames = max(0, int(max_missing_frames))
        self._frame_index = 0
        self._missing_frames = 0
        self._last_box = None

    def is_ready(self):
        return self.use_cascade or hasattr(cv, 'cvtColor')

    def _detect(self, frame):
        if self.use_cascade and self.cascade is not None:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_size,
            )

            if len(faces) == 0:
                return None

            x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
            
            # Eye detection in the top half of the face
            roi_gray = gray[y:y+int(height*0.55), x:x+width]
            right_eye_box = None
            if self.eye_cascade is not None and not self.eye_cascade.empty():
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray, 
                    scaleFactor=1.1, 
                    minNeighbors=4, 
                    minSize=(int(width*0.15), int(height*0.15))
                )
                if len(eyes) > 0:
                    # In a mirrored frame, the user's right eye is on the right side (highest X)
                    ex, ey, ew, eh = max(eyes, key=lambda item: item[0])
                    right_eye_box = FaceBox(int(x + ex), int(y + ey), int(ew), int(eh))
                    
            return FaceBox(int(x), int(y), int(width), int(height), right_eye=right_eye_box)

        return self._detect_skin_region(frame)

    def _detect_skin_region(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)

        skin_hsv = cv.inRange(hsv, (0, 30, 60), (35, 180, 255))
        skin_ycrcb = cv.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
        mask = cv.bitwise_and(skin_hsv, skin_ycrcb)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

        contours_info = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        if not contours:
            return None

        contour = max(contours, key=cv.contourArea)
        area = cv.contourArea(contour)
        if area < 2000:
            return None

        x, y, width, height = cv.boundingRect(contour)
        if width < self.min_size[0] or height < self.min_size[1]:
            return None

        return FaceBox(int(x), int(y), int(width), int(height))

    def _smooth(self, current_box):
        if current_box is None:
            return None

        if self._last_box is None:
            self._last_box = current_box
            return current_box

        smoothed = FaceBox(
            x=int(self._last_box.x * self.smoothing + current_box.x * (1.0 - self.smoothing)),
            y=int(self._last_box.y * self.smoothing + current_box.y * (1.0 - self.smoothing)),
            width=int(self._last_box.width * self.smoothing + current_box.width * (1.0 - self.smoothing)),
            height=int(self._last_box.height * self.smoothing + current_box.height * (1.0 - self.smoothing)),
            right_eye=current_box.right_eye
        )
        self._last_box = smoothed
        return smoothed

    def update(self, frame, force_detect=False):
        if frame is None or not self.is_ready():
            return None

        self._frame_index += 1
        should_detect = force_detect or self._last_box is None or self._frame_index % self.detection_interval == 0

        if should_detect:
            detected = self._detect(frame)
            if detected is None:
                self._missing_frames += 1
                if self._missing_frames > self.max_missing_frames:
                    self._last_box = None
                    return None

                return self._last_box

            self._missing_frames = 0
            return self._smooth(detected)

        return self._last_box

    def face_bounds_with_padding(self, face_box, padding_x=0.18, padding_y=0.28):
        if face_box is None:
            return None

        left = int(face_box.x - face_box.width * padding_x)
        top = int(face_box.y - face_box.height * padding_y)
        width = int(face_box.width * (1.0 + padding_x * 2.0))
        height = int(face_box.height * (1.0 + padding_y * 2.0))
        return left, top, width, height

    def reset(self):
        self._frame_index = 0
        self._missing_frames = 0
        self._last_box = None