import queue
import sounddevice as sd
import json

from vosk import Model , KaldiRecognizer

def start_voice(command_queue):

    MODEL_PATH = r"D:\dev\Jarvis Online\models\vosk-model-small-en-in-0.4"

    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, 16000)

    print('Model loaded Successfully!')
    print("Recognizer ready!")

    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        
        q.put(bytes(indata))


    with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=callback):
        
        print('\nListening\n')

        while True:
            data = q.get()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())

                text = result["text"].strip().lower()

                if text != "":
                    command_queue.put(text)
                    print(f"You said: '{text}'")

                    # if text == "stop" or text == " stop" or text == "stop " or text == " stop ":
                    #     print('GoodBye')
                    #     break
                    if "stop" in text:
                        print('GoodBye')
                        break
