from ctypes import *
import speech_recognition as sr
from speech_to_text import SpeechRecognition
from text_to_speech import TextToSpeach
from brain import JARVIS

def py_error_handler(filename, line, function, err, fmt):
    """Disabling audio warnings
        Credit to "https://blog.yjl.im/2012/11/pyaudio-portaudio-and-alsa-messages.html" for solving the warnings
    """
    pass

LISTEN = True

def main():
    """
    Main loop waiting for speech input.
    """
    speech_recognition = SpeechRecognition()
    recognizer = sr.Recognizer()
    jarvis = JARVIS()
    text_to_speach = TextToSpeach()
    config = {"configurable": {"thread_id": "tomas_test"}}


    while(True):
        try:
            print("J.A.R.V.I.S. started")
            if LISTEN:
                user_input = speech_recognition.transcribe_input(recognizer)
                print("Received user input: ",user_input)
                response = jarvis.send_prompt(
                    messages = user_input, config = config
                )
                
                text_to_speach.speak(response)
        except sr.RequestError as e:
            print(e)
            
        except sr.UnknownValueError as e:
            print(e)

if __name__ == "__main__":
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    main()
