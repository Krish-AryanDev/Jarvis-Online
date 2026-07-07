from pathlib import Path
import time

import cv2 as cv

from Commands import resolve_command
from face_tracking import FaceTracker
from hud import HUDController

def start_camera(command_queue):
    camera = cv.VideoCapture(0)

    if not camera.isOpened():
        print('Unable to turn on camera!')
        return
    
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  #width of the frame
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)  #height of the frame

    face_tracker = FaceTracker()
    hud = HUDController()
    command = ""
    hud_enabled = True
    hud.config.enabled = False
    face_tracking_enabled = False
    diagnostics_enabled = False
    night_vision_enabled = False
    thermal_mode_enabled = False
    recording_enabled = False
    recorder = None
    screenshot_index = 1
    last_frame_time = time.time()
    should_run = True
    screenshot_dir = Path('Images') / 'screenshots'
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    def start_recording(frame):
        nonlocal recorder, recording_enabled
        if recording_enabled:
            return

        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_path = Path('Images') / f'jarvis_recording_{int(time.time())}.avi'
        recorder = cv.VideoWriter(str(output_path), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        recording_enabled = True
        hud.notify('RECORDING STARTED')

    def stop_recording():
        nonlocal recorder, recording_enabled
        if recorder is not None:
            recorder.release()
            recorder = None

        if recording_enabled:
            hud.notify('RECORDING STOPPED')

        recording_enabled = False

    def save_screenshot(frame):
        nonlocal screenshot_index
        file_path = screenshot_dir / f'screenshot_{screenshot_index:03d}.png'
        cv.imwrite(str(file_path), frame)
        screenshot_index += 1
        hud.notify(f'SCREENSHOT SAVED {file_path.name}')

    def apply_night_vision(frame):
        green = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        green = cv.equalizeHist(green)
        green = cv.cvtColor(green, cv.COLOR_GRAY2BGR)
        green[:, :, 0] = (green[:, :, 0] * 0.10).astype('uint8')
        green[:, :, 1] = (green[:, :, 1] * 1.20).clip(0, 255).astype('uint8')
        green[:, :, 2] = (green[:, :, 2] * 0.10).astype('uint8')
        return green

    def apply_thermal_mode(frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return cv.applyColorMap(gray, cv.COLORMAP_JET)

    def handle_command(text, frame):
        nonlocal hud_enabled, face_tracking_enabled, diagnostics_enabled, night_vision_enabled, thermal_mode_enabled, should_run
        resolved = resolve_command(text)
        if resolved is None:
            return

        if resolved.name == 'jarvis':
            hud.notify('At your Service sir', duration=4.0, color=(0, 255, 0))
        elif resolved.name == 'jarvis boot up':
            hud.config.enabled = True
            face_tracking_enabled = True
            diagnostics_enabled = True
            hud.config.show_diagnostics = True
            hud.trigger_boot_sequence()
            hud.notify('SYSTEM BOOT SEQUENCE INITIATED', duration=5.0)
        elif resolved.name == 'enable_hud':
            hud_enabled = True
            hud.config.enabled = True
            hud.notify('HUD ENABLED')
        elif resolved.name == 'disable_hud':
            hud_enabled = False
            hud.config.enabled = False
        elif resolved.name == 'toggle_hud':
            hud_enabled = not hud_enabled
            hud.config.enabled = hud_enabled
            hud.notify('HUD TOGGLED')
        elif resolved.name == 'scan_face':
            face_tracking_enabled = True
            hud.notify('FACE SCAN ACTIVE')
        elif resolved.name == 'night_vision':
            night_vision_enabled = not night_vision_enabled
            hud.notify('NIGHT VISION TOGGLED')
        elif resolved.name == 'thermal_mode':
            thermal_mode_enabled = not thermal_mode_enabled
            hud.notify('THERMAL MODE TOGGLED')
        elif resolved.name == 'start_recording':
            start_recording(frame)
        elif resolved.name == 'stop_recording':
            stop_recording()
        elif resolved.name == 'take_screenshot':
            save_screenshot(frame)
        elif resolved.name == 'toggle_face_tracking':
            face_tracking_enabled = not face_tracking_enabled
            if not face_tracking_enabled:
                face_tracker.reset()
            hud.notify('FACE TRACKING TOGGLED')
        elif resolved.name == 'toggle_diagnostics':
            diagnostics_enabled = not diagnostics_enabled
            hud.config.show_diagnostics = diagnostics_enabled
            hud.notify('DIAGNOSTICS TOGGLED')
        elif resolved.name == 'stop':
            stop_recording()
            should_run = False

    command_pending = None
    
    while should_run:
        success, frame = camera.read()

        if not success:
            print('Camera is not able to Process frames')
            break
        
        while not command_queue.empty():
            command = command_queue.get()
            print(command)
            handle_command(command, frame)

        if not should_run:
            break

        if command == "black":
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        frame = cv.flip(frame, 1)
        display = frame.copy()

        face_box = None
        right_eye_box = None
        if face_tracking_enabled:
            tracked_face = face_tracker.update(display)
            if tracked_face is not None:
                face_box = face_tracker.face_bounds_with_padding(tracked_face)
                if tracked_face.right_eye is not None:
                    re = tracked_face.right_eye
                    right_eye_box = (re.x, re.y, re.width, re.height)

        current_time = time.time()
        elapsed = current_time - last_frame_time
        last_frame_time = current_time
        fps = 0.0 if elapsed <= 0 else 1.0 / elapsed

        if night_vision_enabled:
            display = apply_night_vision(display)

        if thermal_mode_enabled:
            display = apply_thermal_mode(display)

        if hud_enabled:
            hud.draw(display, face_box=face_box, right_eye_box=right_eye_box, fps=fps, battery_level=None)

        if recorder is not None:
            recorder.write(display)

        cv.imshow('Video', display)

        if cv.waitKey(1) & 0xFF == ord('q'):
            print('Camera Closed Successfully!')
            break

    stop_recording()
    camera.release()
    cv.destroyAllWindows()




