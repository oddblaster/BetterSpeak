from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io


load_dotenv()
st.set_page_config(
    page_title="Recorder",
    page_icon="📹",
)
st.sidebar.title("BetterSpeak")

st.title("Audio Analysis")
st.image("Assets/BetterSpeak.png")


api_key = os.getenv("OPENAI_API_KEY")

#Gets the client of OpenAI
client = OpenAI(api_key=api_key)

#Opens the audio file
audio_file= open("Assets/output.wav", "rb")

#Transcribes the audio
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)

#Prints the transcription
st.write(transcription.text)

def load_audio(file):
    y, sr = librosa.load(file)
    return y, sr

def plot_volume(y, sr):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Volume Display')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    return plt

def plot_pitch(y, sr):
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    plt.figure(figsize=(12, 4))
    plt.imshow(pitches, aspect='auto', origin='lower')
    plt.title('Pitch Display')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f')
    return plt

def plot_silent_moments(y, sr):
    rms = librosa.feature.rms(y=y)
    times = librosa.times_like(rms)
    plt.figure(figsize=(12, 4))
    plt.plot(times, rms[0])
    plt.axhline(y=0.1, color='r', linestyle='--')
    plt.title('Silent or Low Volume Moments')
    plt.xlabel('Time')
    plt.ylabel('RMS Energy')
    return plt

def plot_waveform(y, sr):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Waveform Data Display')
    return plt

st.title('Audio File Visualizations')


st.audio(audio_file)

y, sr = load_audio("Assets/output.wav")

st.subheader('1. Volume Display')
volume_plot = plot_volume(y, sr)
st.pyplot(volume_plot)

st.subheader('2. Pitch Display')
pitch_plot = plot_pitch(y, sr)
st.pyplot(pitch_plot)

st.subheader('3. Silent or Low Volume Moments')
silent_plot = plot_silent_moments(y, sr)
st.pyplot(silent_plot)

st.subheader('4. Waveform Data Display')
waveform_plot = plot_waveform(y, sr)
st.pyplot(waveform_plot)




class Score(BaseModel):
    score: int
    name: str
    advice: str

def analyze_candidate(transcription):
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",  # Use the latest available model
        messages=[
            {"role": "system", "content": "You are a recruiter for a software company. You need to score the candidate's performance in the conversation, provide their name, and give advice on how to improve."},
            {"role": "user", "content": f"Here's the transcript of the interview: {transcription}"}
        ],
        functions=[{
            "name": "get_candidate_score",
            "description": "Get the candidate's score, name, and advice",
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "description": "The candidate's score between 0 and 100"
                    },
                    "name": {
                        "type": "string",
                        "description": "The name of the candidate"
                    },
                    "advice": {
                        "type": "string",
                        "description": "Advice for the candidate on how to improve their performance"
                    }
                },
                "required": ["score", "name", "advice"]
            }
        }],
        function_call={"name": "get_candidate_score"}
    )

    result = completion.choices[0].message.function_call.arguments
    return Score.parse_raw(result)

# Usage
st.header("Candidate Analysis")
event = analyze_candidate(transcription)
st.write(f"Score: {event.score}")
st.write(f"Name: {event.name}")
st.write(f"Advice: {event.advice}")