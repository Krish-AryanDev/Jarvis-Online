from camera import start_camera
from voice import start_voice

import threading
import queue

command_queue = queue.Queue()

voice_thread = threading.Thread(target = start_voice, args = (command_queue,))

voice_thread.start()

start_camera(command_queue)

voice_thread.join()