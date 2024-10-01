"""Speech to text module for Jarvis"""
import speech_recognition as sr


class Speech_Recognition:
    def __init__(self) -> None:
        pass

    def transcribe_input(self, speech_recognizer):
        with sr.Microphone(device_index = 0) as source2:
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level 
            speech_recognizer.adjust_for_ambient_noise(source2, duration=1)
            
            #listens for the user's input 
            audio2 = speech_recognizer.listen(source2)
            
            #saving as file for debugging
            #with open("audio_file.wav", "wb") as file:
            #    file.write(audio2.get_wav_data())

            # Using google to recognize audio
            MyText = speech_recognizer.recognize_google(audio2)
            MyText = MyText.lower()
        return MyText
