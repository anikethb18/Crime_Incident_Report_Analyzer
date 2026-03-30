AUDIO SAMPLES — README
======================

This folder is for audio input files (.mp3, .wav, .m4a, .ogg, .flac, .webm).

The audio module (src/audio_transcriber/transcriber.py) processes real audio files.
To test it, place an audio file in this folder and run:

    python run_pipeline.py --audio data/samples/audio_samples/your_file.mp3

WHAT HAPPENS:
  1. OpenAI Whisper API (whisper-1) transcribes speech to text
  2. If API unavailable, local whisper.load_model("base") is used as fallback
  3. GPT-4o extracts structured incident fields from the transcript
  4. Returns: incident_id, date, time, location, incident_type, suspects, etc.

FREE SOURCES FOR TEST AUDIO:
  - Generate a short 911-style clip with any text-to-speech tool (e.g. gTTS, ElevenLabs)
  - Record yourself reading one of the transcripts in data/samples/text_samples/
  - LibriVox public domain recordings: https://librivox.org

GENERATE A SAMPLE WITH gTTS (Python):
    pip install gTTS
    python -c "
    from gtts import gTTS
    text = '''
    Yes hello, there is a fight outside Club Luxe on Westheimer and Montrose.
    Two guys, one hit the other with a glass bottle. There is blood everywhere.
    Please hurry.
    '''
    gTTS(text, lang='en').save('data/samples/audio_samples/sample_911_call.mp3')
    print('saved')
    "
    python run_pipeline.py --audio data/samples/audio_samples/sample_911_call.mp3

SYSTEM REQUIREMENTS (for local Whisper fallback):
    pip install openai-whisper torch
    python run_pipeline.py --audio your_file.mp3 --local-whisper
