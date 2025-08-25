import streamlit as st
import torch
import os
import tempfile
from nemo.collections.asr.models import ASRModel
from pydub import AudioSegment
import datetime
import pandas as pd
import time
import gc

# Set page config with a modern theme
st.set_page_config(
    page_title="Parakeet ASR Demo",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        padding: 1rem 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        font-size: 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1E88E5;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: -1rem 0 1rem 0 !important;
    }
    h3 {
        color: #1E88E5;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .info-box {
        background-color: rgba(30, 136, 229, 0.1);
        border-left: 5px solid #1E88E5;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .success-box {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 5px solid #4CAF50;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        line-height: 1.2;
    }
    .warning-box {
        background-color: rgba(255, 152, 0, 0.1);
        border-left: 5px solid #FF9800;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .error-box {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 5px solid #F44336;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_FORMATS = ['wav', 'flac']
MAX_RECOMMENDED_DURATION = 30 * 60  # 30 minutes in seconds
LONG_AUDIO_THRESHOLD = 480  # 8 minutes in seconds

# Initialize session state variables
if 'transcription_history' not in st.session_state:
    st.session_state.transcription_history = []

def custom_info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def custom_success(text):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def custom_warning(text):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)

def custom_error(text):
    st.markdown(f'<div class="error-box">{text}</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the ASR model with proper error handling and visualization."""
    try:
        with st.spinner("Loading Parakeet TDT model... This may take a minute."):
            model = ASRModel.from_pretrained(model_name=MODEL_NAME)
            model.eval()
            
            # Move to appropriate device
            model = model.to(DEVICE)
            
            # Disabled according to https://github.com/SridharSampath/parakeet-asr-demo/issues/2
            # # Use mixed precision for better performance when using GPU
            # if DEVICE == "cuda":
            #     model = model.to(torch.bfloat16)
                
            custom_success(f"Model loaded successfully on {DEVICE.upper()}!")
            return model
    except Exception as e:
        custom_error(f"Error loading model: {str(e)}")
        if "CUDA" in str(e) or "GPU" in str(e):
            custom_info("GPU issues detected. Try running with CPU by setting device to 'cpu'.")
        return None

def process_audio(audio_path):
    """Process audio file for transcription with progress reporting."""
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        duration_sec = audio.duration_seconds
        
        if duration_sec > MAX_RECOMMENDED_DURATION:
            custom_warning(f"Audio is very long ({duration_sec/60:.1f} minutes). Transcription may take a while and could encounter memory issues.")
        
        # Progress reporting
        progress_bar = st.progress(0)
        
        
        # Resample to 16kHz if needed
        if audio.frame_rate != 16000:
            progress_bar.progress(0.2)
            audio = audio.set_frame_rate(16000)
        
        progress_bar.progress(0.5)
        
        # Convert to mono if stereo
        if audio.channels > 1:            
            audio = audio.set_channels(1)
        
        progress_bar.progress(0.8)
        
        # Save processed audio
        temp_dir = tempfile.gettempdir()
        processed_path = os.path.join(temp_dir, "processed_audio.wav")
        audio.export(processed_path, format="wav")
        
        progress_bar.progress(1.0)
        custom_success("Audio processed successfully!")
        
        return processed_path, duration_sec
    except Exception as e:
        custom_error(f"Error processing audio: {str(e)}")
        
        # Provide more specific error messages based on common issues
        if "No such file" in str(e):
            custom_info("The audio file could not be found. Please upload it again.")
        elif "Unsupported format" in str(e) or "unknown format" in str(e):
            custom_info(f"File format not supported. Please upload one of these formats: {', '.join(SUPPORTED_FORMATS)}")
        elif "memory" in str(e).lower():
            custom_info("Memory error occurred. Try with a shorter audio file or restart the application.")
            
        return None, None

def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=seconds)).split('.')[0]

def transcribe_audio(audio_path, show_progress=True):
    """Transcribe audio file using the model with detailed progress reporting."""
    start_time = time.time()
    try:
        model = load_model()
        if model is None:
            return None
        
        processed_path, duration_sec = process_audio(audio_path)
        if processed_path is None:
            return None
        
        # Apply long audio settings if needed
        long_audio_settings_applied = False
        if duration_sec > LONG_AUDIO_THRESHOLD:
            try:
                custom_info(f"Audio longer than {LONG_AUDIO_THRESHOLD/60:.1f} minutes. Applying optimized settings for long transcription.")
                model.change_attention_model("rel_pos_local_attn", [256, 256])
                model.change_subsampling_conv_chunking_factor(1)
                long_audio_settings_applied = True
            except Exception as e:
                custom_warning(f"Could not apply long audio settings: {str(e)}")
        
        try:
            if show_progress:
                with st.spinner("Transcribing audio... This may take a while for longer files."):
                    progress_bar = st.progress(0)
                    
                    # Create a progress indicator that updates based on estimated time
                    # This is an estimation since we can't track actual progress
                    estimated_total_time = duration_sec * 0.5  # Rough estimate: processing takes ~50% of audio duration
                    
                    def update_progress():
                        for i in range(1, 101):
                            elapsed = time.time() - start_time
                            if elapsed >= estimated_total_time:
                                break
                            # progress = min(elapsed / estimated_total_time, 0.95)  # Max at 95% until completion
                            # progress_bar.progress(progress)
                            time.sleep(estimated_total_time / 100)
                    
                    # Start progress update in separate thread
                    import threading
                    progress_thread = threading.Thread(target=update_progress)
                    progress_thread.daemon = True
                    progress_thread.start()
                    
                    # Actual transcription
                    output = model.transcribe([processed_path], timestamps=True)
                    
                    # Complete progress
                    progress_bar.progress(1.0)
            else:
                output = model.transcribe([processed_path], timestamps=True)
            
            if not output or not isinstance(output, list) or not output[0] or not hasattr(output[0], 'timestamp'):
                custom_error("Transcription failed or produced unexpected output format.")
                return None
            
            segment_timestamps = output[0].timestamp['segment']
            
            # Generate CSV content with better headers
            csv_data = [["From (s)", "To (s)", "From (time)", "To (time)", "Duration", "Transcription"]]
            
            for ts in segment_timestamps:
                start_s = ts['start']
                end_s = ts['end']
                start_formatted = format_time(start_s)
                end_formatted = format_time(end_s)
                duration = end_s - start_s
                
                csv_data.append([
                    f"{start_s:.2f}", 
                    f"{end_s:.2f}", 
                    start_formatted,
                    end_formatted,
                    f"{duration:.2f}",
                    ts['segment']
                ])
            
            processing_time = time.time() - start_time
            custom_success(f"Transcription completed in {processing_time:.1f} seconds!")
            
            # Save to history
            filename = os.path.basename(audio_path)
            st.session_state.transcription_history.append({
                'filename': filename,
                'duration': duration_sec,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'csv_data': csv_data
            })
            
            return csv_data
            
        finally:
            # Cleanup
            if long_audio_settings_applied:
                try:
                    model.change_attention_model("rel_pos")
                    model.change_subsampling_conv_chunking_factor(-1)
                except Exception as e:
                    custom_warning(f"Issue reverting model settings: {str(e)}")
            
            if os.path.exists(processed_path):
                os.remove(processed_path)
            
            # Force garbage collection
            torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        custom_error(f"Error during transcription: {str(e)}")
        
        # Offer more helpful advice based on error
        if "CUDA out of memory" in str(e):
            custom_info("GPU ran out of memory. Try processing a shorter audio file, or restart the application.")
        elif "timeout" in str(e).lower():
            custom_info("The operation timed out. This could be due to the file size or server load.")
        
        return None

def export_to_formats(csv_data):
    """Create exportable data in multiple formats"""
    # For CSV
    csv_string = "\n".join([",".join([f'"{cell}"' if ',' in cell else cell for cell in row]) for row in csv_data])
    
    # For plain text (just the transcription)
    text_string = "\n\n".join([row[5] for row in csv_data[1:]])
    
    # For SRT (subtitle format)
    srt_string = ""
    for i, row in enumerate(csv_data[1:], 1):
        start_s = float(row[0])
        end_s = float(row[1])
        
        # Convert to SRT time format (HH:MM:SS,mmm)
        start_srt = f"{int(start_s//3600):02d}:{int((start_s%3600)//60):02d}:{int(start_s%60):02d},{int((start_s%1)*1000):03d}"
        end_srt = f"{int(end_s//3600):02d}:{int((end_s%3600)//60):02d}:{int(end_s%60):02d},{int((end_s%1)*1000):03d}"
        
        srt_string += f"{i}\n{start_srt} --> {end_srt}\n{row[5]}\n\n"
    
    return csv_string, text_string, srt_string

# Main UI
st.title("🎙️ Speech Transcription with Parakeet TDT")

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Transcribe", "About"])

# Main content based on selected page
if page == "Transcribe":
    # Description
    st.markdown("""
    This demo showcases [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), a 600-million-parameter model designed for high-quality English speech recognition.

    **Key Features:**
    - Automatic punctuation and capitalization
    - Accurate word-level timestamps
    - Efficiently transcribes long audio segments
    - Robust performance on spoken numbers and song lyrics transcription
    """)

    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploader with expanded format support
        uploaded_file = st.file_uploader("Upload an audio file (WAV or FLAC)", 
                                        type=SUPPORTED_FORMATS)

    with col2:
        if uploaded_file:
            st.markdown("### Audio Preview")
            st.audio(uploaded_file)
            
            # Display basic audio info
            file_info = f"**File:** {uploaded_file.name}<br>"
            file_info += f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB<br>"
            st.markdown(f'<div class="info-box">{file_info}</div>', unsafe_allow_html=True)

    if uploaded_file:
        # Save uploaded file
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🎯 Transcribe Audio", type="primary"):
            with st.spinner("Processing and transcribing audio..."):
                csv_data = transcribe_audio(audio_path)
                
                if csv_data:
                    # Display transcription results
                    st.markdown("### 📝 Transcription Results")
                    
                    # Create a dataframe for display
                    df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
                    
                    # Display as table with modern styling
                    st.dataframe(
                        df,
                        column_config={
                            "From (s)": st.column_config.NumberColumn(format="%.2f", width="small"),
                            "To (s)": st.column_config.NumberColumn(format="%.2f", width="small"),
                            "From (time)": st.column_config.TextColumn(width="small"),
                            "To (time)": st.column_config.TextColumn(width="small"),
                            "Duration": st.column_config.NumberColumn(format="%.2f", width="small"),
                            "Transcription": st.column_config.TextColumn(width="large")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Create export strings for different formats
                    csv_string, text_string, srt_string = export_to_formats(csv_data)
                    
                    # Show export options
                    st.markdown("### 📥 Export Options")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            "📄 Download as CSV",
                            data=csv_string,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        st.download_button(
                            "📝 Download as Text",
                            data=text_string,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                    with col3:
                        st.download_button(
                            "🎬 Download as SRT",
                            data=srt_string,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_subtitle.srt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    # Word count analysis
                    total_words = sum(len(row[5].split()) for row in csv_data[1:])
                    total_duration = float(csv_data[-1][1]) - float(csv_data[1][0])
                    words_per_minute = (total_words / total_duration) * 60 if total_duration > 0 else 0
                    
                    st.markdown("### 📊 Analysis")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Words", f"{total_words}")
                    col2.metric("Speech Duration", f"{format_time(total_duration)}")
                    col3.metric("Words per Minute", f"{words_per_minute:.1f}")
            
            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)

elif page == "About":
    st.title("About this Application")
    
    st.markdown("""
    ## Parakeet Speech Recognition
    
    This application uses NVIDIA's Parakeet-TDT, a powerful speech recognition model designed for accurate transcription with timestamps and punctuation.
    
    ### Model Details
    
    - **Model**: [nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
    - **Parameters**: 600 million
    - **Features**: Automatic punctuation, capitalization, and word-level timestamps
    - **Language**: English
    
   
    ### About Speech Recognition Technology
    
    Modern speech recognition systems like Parakeet use advanced neural networks trained on thousands of hours of speech data. These systems analyze audio waveforms to predict the most likely sequence of words being spoken, taking into account language patterns and context.
    """)
    
    # System information
    st.subheader("System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**System Platform:** {os.name.upper()}")
        st.markdown(f"**PyTorch Version:** {torch.__version__}")
    
    with col2:
        if torch.cuda.is_available():
            st.markdown(f"**CUDA Available:** Yes (Version {torch.version.cuda})")
            st.markdown(f"**GPU:** {torch.cuda.get_device_name(0)}")
        else:
            st.markdown("**CUDA Available:** No (Using CPU)")

# Footer
st.markdown("---")
st.markdown("Made with NVIDIA's Parakeet-TDT model")