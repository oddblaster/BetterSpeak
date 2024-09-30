
import streamlit as st
import cv2
import pyaudio
import wave
import time

st.set_page_config(
    page_title="Recorder",
    page_icon="📹",
)
st.sidebar.title("BetterSpeak")
# Initialize session state using get() method
is_recording = st.session_state.get('is_recording', False)
frames = st.session_state.get('frames', [])
audio_thread = st.session_state.get('audio_thread', None)

#Variables for audio recording

#Number of audio frames that are processed at a time during recording
CHUNK = 1024

#Format of the audio data
FORMAT = pyaudio.paInt16

#Number of audio channels
CHANNELS = 1

#Sample rate of the audio data
RATE = 44100

#Number of seconds to record
RECORD_SECONDS = 15

st.title("Speech")
st.image("Assets/BetterSpeak.png")

# Initialize session state
if 'frames' not in st.session_state:
    st.session_state.frames = []
if 'recording' not in st.session_state:
    st.session_state.recording = False

# UI elements
col1, col2 = st.columns(2)
start_button = col1.button("Start Recording")
stop_button = col2.button("Stop Recording")

frame_placeholder = st.empty()
status_placeholder = st.empty()

#Function to record audio and video
def record_audio_and_video():
    #Initialize PyAudio
    p = pyaudio.PyAudio()
    
    #Open a stream to capture audio data
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    #Initialize video capture
    cap = cv2.VideoCapture(0)
    
    #Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("Assets/video_output.mp4", fourcc, 20.0, (640, 480))
    
    #Initialize frames and start time
    st.session_state.frames = []
    start_time = time.time()
    
    #Loop to record audio and video
    while st.session_state.recording and (time.time() - start_time) < RECORD_SECONDS:
        # Audio recording
        audio_data = stream.read(CHUNK)
        st.session_state.frames.append(audio_data)
        
        # Video recording
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        
        # Display video
        frame_placeholder.image(frame, channels="BGR")
        
        # Update status
        # Update status
        elapsed_time = time.time() - start_time
        status_placeholder.text(f"Recording: {elapsed_time:.2f} / {RECORD_SECONDS} seconds")
    
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    cap.release()
    out.release()
    
    # Save audio to wav file
    wf = wave.open("Assets/output.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(st.session_state.frames))
    wf.close()
    
    #Set recording to false
    st.session_state.recording = False
    
    #Update status
    status_placeholder.text("Recording completed and saved.")

if start_button:
    st.session_state.recording = True
    record_audio_and_video()

if stop_button:
    st.session_state.recording = False

status_placeholder.text("Recording status: " + ("Recording" if st.session_state.recording else "Not recording"))