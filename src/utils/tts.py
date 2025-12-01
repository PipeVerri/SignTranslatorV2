from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

def speak(text, lang='en'):
    """Reproducir texto usando gTTS y pydub"""
    # Crear el audio en memoria (sin archivo)
    mp3_fp = BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # Cargar y reproducir
    audio = AudioSegment.from_mp3(mp3_fp)
    play(audio)

if __name__ == "__main__":
    speak("Hello world")