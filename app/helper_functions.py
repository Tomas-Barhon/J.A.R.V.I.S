def save_audio_file(audio):
    """
    Saves 
    
    """
    with open("audio_file.wav", "wb") as file:
        file.write(audio.get_wav_data())