import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print('Talk biiitch.')
    audio_text = r.listen(source, timeout=10)
    print('Speech detected.')

text = r.recognize_google(audio_text, show_all=False)
print('Transcription: ' + text)
