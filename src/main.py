from camera import start_camera
from voice import start_voice

import threading

voice_thread = threading.Thread(target = start_voice)

voice_thread.start()

start_camera()

voice_thread.join()