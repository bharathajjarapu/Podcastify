import streamlit as st
import torch
from groq import Groq
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import numpy as np
from scipy.io import wavfile
import io
import os
from dotenv import load_dotenv

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load Parler-TTS models and tokenizers
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Descriptions for each speaker
speaker1_description = "Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with very close recording."
speaker2_description = "Jon's voice is calm, slightly fast in delivery, with very close recording and almost no background noise."

def generate_text_with_groq(prompt):
    """Generate podcast script using Groq API."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant for generating podcast scripts."},
        {"role": "user", "content": prompt}
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        response_format={"type": "json_object"}
    )
    return chat_completion.choices[0].message.content

def generate_audio(text, description):
    """Generate audio using Parler-TTS for a specified speaker."""
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, model.config.sampling_rate

def numpy_to_audio_buffer(audio_arr, sampling_rate):
    """Convert numpy array to a byte buffer compatible with Streamlit."""
    audio_int16 = (audio_arr * 32767).astype("int16")
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)
    return byte_io

# Streamlit interface
st.title("Podcast Generator using Groq and Parler-TTS")
st.write("Generate a podcast script using Groq and convert it to audio using Parler-TTS.")

# User input for podcast topic
podcast_topic = st.text_input("Enter podcast topic or prompt", value="Generate a podcast script on AI advancements.")

if st.button("Generate Podcast"):
    st.write("Generating podcast script...")
    
    # Generate text using Groq
    podcast_text = generate_text_with_groq(podcast_topic)
    st.write("Generated Podcast Text:")
    st.write(podcast_text)

    # Convert podcast text into speaker-specific text segments
    podcast_segments = [
        ("Speaker 1", "Welcome to this week's episode of AI Insights!"),
        ("Speaker 2", "Hi there! Today, we are diving into AI advancements."),
        ("Speaker 1", podcast_text)
    ]

    # Generate and concatenate audio for each segment
    final_audio_buffer = io.BytesIO()
    for speaker, text in podcast_segments:
        description = speaker1_description if speaker == "Speaker 1" else speaker2_description
        audio_arr, rate = generate_audio(text, description)
        audio_segment = numpy_to_audio_buffer(audio_arr, rate)
        
        final_audio_buffer.write(audio_segment.read())

    final_audio_buffer.seek(0)

    # Streamlit audio player for final podcast
    st.audio(final_audio_buffer, format="audio/wav", start_time=0)
    st.download_button(label="Download Podcast", data=final_audio_buffer, file_name="podcast.wav", mime="audio/wav")
