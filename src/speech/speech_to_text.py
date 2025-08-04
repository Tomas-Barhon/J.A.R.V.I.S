"""Speech to text module for Jarvis"""
import speech_recognition as sr
from src.helpers.helper_functions import save_audio_file

class SpeechRecognition:
    def __init__(self, debug = False) -> None:
        self.debug = debug

    def transcribe_input(self, speech_recognizer):
        with sr.Microphone(device_index = 0) as source2:
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            speech_recognizer.adjust_for_ambient_noise(source2, duration=1)

            #listens for the user's input 
            audio2 = speech_recognizer.listen(source2)

            #if debug save file
            if self.debug:
                save_audio_file(audio2)

            # Using google recognizer for now
            MyText = speech_recognizer.recognize_google(audio2)
            MyText = MyText.lower()
        return MyText
