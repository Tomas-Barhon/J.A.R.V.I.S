import speech_recognition as sr
from ctypes import *
from speech_to_text import Speech_Recognition

#Disabling audio warnings
#Credit to "https://blog.yjl.im/2012/11/pyaudio-portaudio-and-alsa-messages.html" for solving the warnings
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so')
asound.snd_lib_error_set_handler(c_error_handler)




speech_recognition = Speech_Recognition()
r = sr.Recognizer()
while(True):
    # Exception handling to handle
    # exceptions at the runtime
    try:
        
        print(speech_recognition.transcribe_input(r))
        
            
    except sr.RequestError as e:
        print(e)
        
    except sr.UnknownValueError as e:
        print(e)