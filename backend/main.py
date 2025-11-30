from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from scipy import signal
from scipy.io import wavfile
import tempfile
import os

app = FastAPI(title="Audio Waveform API")

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Audio Waveform API is running"}


def load_audio_file(file_path: str):
    """Load audio file and return samples and sample rate"""
    ext = os.path.splitext(file_path)[1].lower()
    
    # Try scipy wavfile first for WAV files
    if ext == '.wav':
        try:
            sample_rate, samples = wavfile.read(file_path)
            # Convert to float and normalize
            if samples.dtype == np.int16:
                samples = samples.astype(np.float32) / 32768.0
            elif samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 2147483648.0
            elif samples.dtype == np.uint8:
                samples = (samples.astype(np.float32) - 128) / 128.0
            elif samples.dtype == np.float32 or samples.dtype == np.float64:
                samples = samples.astype(np.float32)
            # Convert stereo to mono
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)
            return samples, sample_rate
        except Exception as e:
            raise ValueError(f"Failed to read WAV file: {str(e)}")
    
    # For other formats, try pydub (requires FFmpeg)
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples = samples / np.iinfo(np.int16).max
        return samples, audio.frame_rate
    except Exception as e:
        raise ValueError(
            f"Failed to decode audio file. For MP3/OGG/M4A files, FFmpeg must be installed. "
            f"Please convert to WAV format or install FFmpeg. Error: {str(e)}"
        )


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...), analysis_type: str = "waveform"):
    """
    Analyze audio file and return visualization data
    
    analysis_type options:
    - waveform: Time domain amplitude
    - fft: Frequency spectrum (magnitude)
    - spectrogram: Time-frequency representation
    - power_spectrum: Power spectral density
    """
    allowed_extensions = ('.mp3', '.wav', '.ogg', '.m4a', '.flac')
    if not file.filename.lower().endswith(allowed_extensions):
        return JSONResponse(
            status_code=400,
            content={"error": f"Please upload an audio file ({', '.join(allowed_extensions)})"}
        )
    
    try:
        # Read and save uploaded file temporarily
        contents = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Load audio
        samples, sample_rate = load_audio_file(tmp_path)
        os.unlink(tmp_path)
        
        duration = len(samples) / sample_rate
        num_samples = len(samples)
        
        result = {
            "filename": file.filename,
            "duration": duration,
            "sample_rate": sample_rate,
            "analysis_type": analysis_type
        }
        
        if analysis_type == "waveform":
            # Downsample waveform for visualization
            target_points = 10000
            step = max(1, num_samples // target_points)
            waveform = samples[::step].tolist()
            result["waveform"] = waveform
            result["points"] = len(waveform)
            
        elif analysis_type == "fft":
            # Compute FFT - frequency spectrum
            n_fft = min(8192, num_samples)
            if n_fft < 256:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Audio file too short for FFT analysis"}
                )
            
            # Use window to reduce spectral leakage
            window = np.hanning(n_fft)
            
            # Take middle section of audio for FFT
            start = max(0, num_samples // 2 - n_fft // 2)
            segment = samples[start:start + n_fft] * window
            
            # Compute FFT
            fft_result = np.fft.rfft(segment)
            magnitudes = np.abs(fft_result)
            
            # Convert to dB scale
            magnitudes_db = 20 * np.log10(magnitudes + 1e-10)
            
            # Frequency bins
            frequencies = np.fft.rfftfreq(n_fft, 1/sample_rate)
            
            # Downsample for visualization (keep up to 20kHz)
            max_freq_idx = min(len(frequencies), np.searchsorted(frequencies, 20000))
            if max_freq_idx < 10:
                max_freq_idx = len(frequencies)
            
            target_points = 2000
            step = max(1, max_freq_idx // target_points)
            
            result["frequencies"] = frequencies[:max_freq_idx:step].tolist()
            result["magnitudes"] = magnitudes_db[:max_freq_idx:step].tolist()
            result["points"] = len(result["frequencies"])
            
        elif analysis_type == "spectrogram":
            # Compute spectrogram
            # Ensure nperseg is valid
            nperseg = min(2048, max(256, num_samples // 16))
            noverlap = nperseg // 2
            
            if num_samples < nperseg * 2:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Audio file too short for spectrogram analysis"}
                )
            
            f, t, Sxx = signal.spectrogram(
                samples, 
                fs=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='spectrum'
            )
            
            # Convert to dB
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            # Limit frequency range to 0-20kHz
            max_freq_idx = min(len(f), np.searchsorted(f, 20000))
            if max_freq_idx < 5:
                max_freq_idx = len(f)
            
            f = f[:max_freq_idx]
            Sxx_db = Sxx_db[:max_freq_idx, :]
            
            # Downsample for reasonable data size
            # Target: ~200 frequency bins, ~500 time bins
            freq_step = max(1, len(f) // 200)
            time_step = max(1, len(t) // 500)
            
            f_downsampled = f[::freq_step]
            t_downsampled = t[::time_step]
            Sxx_downsampled = Sxx_db[::freq_step, ::time_step]
            
            result["frequencies"] = f_downsampled.tolist()
            result["times"] = t_downsampled.tolist()
            result["spectrogram"] = Sxx_downsampled.tolist()
            result["min_db"] = float(np.min(Sxx_downsampled))
            result["max_db"] = float(np.max(Sxx_downsampled))
            
        elif analysis_type == "power_spectrum":
            # Compute Power Spectral Density using Welch's method
            nperseg = min(4096, max(256, num_samples // 8))
            
            if num_samples < nperseg * 2:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Audio file too short for power spectrum analysis"}
                )
            
            f, Pxx = signal.welch(
                samples,
                fs=sample_rate,
                nperseg=nperseg,
                scaling='density'
            )
            
            # Convert to dB
            Pxx_db = 10 * np.log10(Pxx + 1e-10)
            
            # Limit to 20kHz
            max_freq_idx = min(len(f), np.searchsorted(f, 20000))
            if max_freq_idx < 10:
                max_freq_idx = len(f)
            
            # Downsample
            target_points = 2000
            step = max(1, max_freq_idx // target_points)
            
            result["frequencies"] = f[:max_freq_idx:step].tolist()
            result["power"] = Pxx_db[:max_freq_idx:step].tolist()
            result["points"] = len(result["frequencies"])
            
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unknown analysis type: {analysis_type}"}
            )
        
        return result
        
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing audio: {str(e)}", "trace": traceback.format_exc()}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
