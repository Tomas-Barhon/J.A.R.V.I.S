def save_audio_file(audio):
    """
    Saves audio data to a file for debugging purposes.
    
    """
    with open("audio_file.wav", "wb") as file:
        file.write(audio.get_wav_data())