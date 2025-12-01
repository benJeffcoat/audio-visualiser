from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from scipy import signal
from scipy.io import wavfile
import tempfile
import os
import json
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass, field
import uuid

app = FastAPI(title="Audio Waveform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store processed audio data in memory (keyed by session_id)
audio_cache: Dict[str, dict] = {}


def load_audio_file(file_path: str):
    """Load audio file and return samples and sample rate"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.wav':
        try:
            sample_rate, samples = wavfile.read(file_path)
            if samples.dtype == np.int16:
                samples = samples.astype(np.float32) / 32768.0
            elif samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 2147483648.0
            elif samples.dtype == np.uint8:
                samples = (samples.astype(np.float32) - 128) / 128.0
            elif samples.dtype in [np.float32, np.float64]:
                samples = samples.astype(np.float32)
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)
            return samples, sample_rate
        except Exception as e:
            raise ValueError(f"Failed to read WAV file: {str(e)}")
    
    # For other formats, use pydub (requires FFmpeg)
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
            f"Failed to decode audio. FFmpeg required for MP3/OGG/M4A. Error: {str(e)}"
        )


def precompute_waveform(samples: np.ndarray, sample_rate: int, target_points: int = 10000):
    """Downsample waveform for visualization"""
    step = max(1, len(samples) // target_points)
    waveform = samples[::step]
    max_amplitude = float(np.max(np.abs(waveform))) * 1.05
    return {
        "waveform": waveform.tolist(),
        "max_amplitude": max(max_amplitude, 0.01),
        "points": len(waveform)
    }


def precompute_fft(samples: np.ndarray, sample_rate: int):
    """Compute FFT frequency spectrum"""
    n_fft = min(8192, len(samples))
    if n_fft < 256:
        return None
    
    window = np.hanning(n_fft)
    start = max(0, len(samples) // 2 - n_fft // 2)
    segment = samples[start:start + n_fft] * window
    
    fft_result = np.fft.rfft(segment)
    magnitudes = np.abs(fft_result)
    magnitudes_db = 20 * np.log10(magnitudes + 1e-10)
    frequencies = np.fft.rfftfreq(n_fft, 1/sample_rate)
    
    # Limit to 20kHz and downsample
    max_freq_idx = min(len(frequencies), np.searchsorted(frequencies, 20000))
    step = max(1, max_freq_idx // 1000)
    
    return {
        "frequencies": frequencies[:max_freq_idx:step].tolist(),
        "magnitudes": magnitudes_db[:max_freq_idx:step].tolist()
    }


def precompute_spectrogram(samples: np.ndarray, sample_rate: int):
    """Compute spectrogram (time-frequency representation)"""
    nperseg = min(2048, max(256, len(samples) // 16))
    noverlap = nperseg // 2
    
    if len(samples) < nperseg * 2:
        return None
    
    f, t, Sxx = signal.spectrogram(
        samples, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, scaling='spectrum'
    )
    
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Limit frequency to 15kHz
    max_freq_idx = min(len(f), np.searchsorted(f, 15000))
    f = f[:max_freq_idx]
    Sxx_db = Sxx_db[:max_freq_idx, :]
    
    # Downsample for reasonable size (~150 freq bins, ~400 time bins)
    freq_step = max(1, len(f) // 150)
    time_step = max(1, len(t) // 400)
    
    return {
        "frequencies": f[::freq_step].tolist(),
        "times": t[::time_step].tolist(),
        "spectrogram": Sxx_db[::freq_step, ::time_step].tolist(),
        "min_db": float(np.min(Sxx_db)),
        "max_db": float(np.max(Sxx_db))
    }


def precompute_power_spectrum(samples: np.ndarray, sample_rate: int):
    """Compute Power Spectral Density using Welch's method"""
    nperseg = min(4096, max(256, len(samples) // 8))
    
    if len(samples) < nperseg * 2:
        return None
    
    f, Pxx = signal.welch(samples, fs=sample_rate, nperseg=nperseg, scaling='density')
    Pxx_db = 10 * np.log10(Pxx + 1e-10)
    
    max_freq_idx = min(len(f), np.searchsorted(f, 20000))
    step = max(1, max_freq_idx // 1000)
    
    return {
        "frequencies": f[:max_freq_idx:step].tolist(),
        "power": Pxx_db[:max_freq_idx:step].tolist()
    }


def bandpass_filter(samples: np.ndarray, sample_rate: int, low_freq: float, high_freq: float):
    """Apply bandpass filter to isolate frequency band"""
    nyquist = sample_rate / 2
    
    # Clamp frequencies to valid range
    low = max(20, low_freq) / nyquist
    high = min(high_freq, nyquist - 100) / nyquist
    
    if low >= high or low >= 1 or high <= 0:
        return np.zeros_like(samples, dtype=np.float32)
    
    # Design butterworth bandpass filter
    try:
        # Use a lower order filter for stability
        order = min(4, max(1, int(len(samples) / 1000)))
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, samples, padlen=min(150, len(samples) - 1))
        
        # Replace any NaN or Inf values with 0
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
        return filtered.astype(np.float32)
    except Exception as e:
        print(f"Bandpass filter error ({low_freq}-{high_freq}Hz): {e}")
        return np.zeros_like(samples, dtype=np.float32)


def precompute_frequency_bands(samples: np.ndarray, sample_rate: int, target_points: int = 10000):
    """
    Separate audio into frequency bands for multi-band visualization.
    Bands: Sub-bass, Bass, Low-mid, Mid, High-mid, High
    """
    # Define frequency bands (in Hz)
    bands = {
        "sub_bass": {"low": 20, "high": 60, "color": "#ef4444", "label": "Sub Bass (20-60 Hz)"},
        "bass": {"low": 60, "high": 250, "color": "#f97316", "label": "Bass (60-250 Hz)"},
        "low_mid": {"low": 250, "high": 500, "color": "#eab308", "label": "Low Mid (250-500 Hz)"},
        "mid": {"low": 500, "high": 2000, "color": "#22c55e", "label": "Mid (500-2k Hz)"},
        "high_mid": {"low": 2000, "high": 6000, "color": "#3b82f6", "label": "High Mid (2k-6k Hz)"},
        "high": {"low": 6000, "high": 20000, "color": "#a855f7", "label": "High (6k-20k Hz)"},
    }
    
    step = max(1, len(samples) // target_points)
    result = {"bands": {}}
    
    # Calculate global max amplitude for normalization
    global_max = float(np.max(np.abs(samples))) * 1.05
    
    for band_name, band_info in bands.items():
        filtered = bandpass_filter(samples, sample_rate, band_info["low"], band_info["high"])
        downsampled = filtered[::step]
        band_max = float(np.max(np.abs(downsampled))) if len(downsampled) > 0 else 0.01
        
        result["bands"][band_name] = {
            "waveform": downsampled.tolist(),
            "max_amplitude": max(band_max, 0.001),
            "color": band_info["color"],
            "label": band_info["label"],
            "low_freq": band_info["low"],
            "high_freq": band_info["high"]
        }
    
    result["global_max_amplitude"] = max(global_max, 0.01)
    result["points"] = len(samples[::step])
    
    return result


def precompute_frequency_band_segments(samples: np.ndarray, sample_rate: int, duration: float):
    """Pre-compute frequency band waveforms for different time windows"""
    bands_config = {
        "sub_bass": {"low": 20, "high": 60, "color": "#ef4444", "label": "Sub Bass"},
        "bass": {"low": 60, "high": 250, "color": "#f97316", "label": "Bass"},
        "low_mid": {"low": 250, "high": 500, "color": "#eab308", "label": "Low Mid"},
        "mid": {"low": 500, "high": 2000, "color": "#22c55e", "label": "Mid"},
        "high_mid": {"low": 2000, "high": 6000, "color": "#3b82f6", "label": "High Mid"},
        "high": {"low": 6000, "high": 20000, "color": "#a855f7", "label": "High"},
    }
    
    # Pre-filter all bands
    filtered_bands = {}
    for band_name, band_info in bands_config.items():
        filtered_bands[band_name] = {
            "samples": bandpass_filter(samples, sample_rate, band_info["low"], band_info["high"]),
            "color": band_info["color"],
            "label": band_info["label"]
        }
    
    segments = {}
    window_sizes = [1, 2, 3, 5, 10]
    
    for window_size in window_sizes:
        if window_size > duration:
            continue
        
        samples_per_window = int(window_size * sample_rate)
        target_points = min(2000, samples_per_window)
        step = max(1, samples_per_window // target_points)
        hop_seconds = window_size / 4
        hop_samples = int(hop_seconds * sample_rate)
        
        band_segments = {band_name: {"segments": [], "color": info["color"], "label": info["label"]} 
                        for band_name, info in filtered_bands.items()}
        start_times = []
        
        for start in range(0, len(samples) - samples_per_window + 1, hop_samples):
            start_times.append(start / sample_rate)
            for band_name, band_info in filtered_bands.items():
                segment = band_info["samples"][start:start + samples_per_window:step]
                # Clean NaN values
                segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
                band_segments[band_name]["segments"].append(segment.tolist())
        
        segments[window_size] = {
            "bands": band_segments,
            "start_times": start_times,
            "points_per_segment": target_points
        }
    
    # Full view
    full_step = max(1, len(samples) // 3000)
    full_bands = {}
    for band_name, band_info in filtered_bands.items():
        full_segment = band_info["samples"][::full_step]
        # Clean NaN values
        full_segment = np.nan_to_num(full_segment, nan=0.0, posinf=0.0, neginf=0.0)
        full_bands[band_name] = {
            "segments": [full_segment.tolist()],
            "color": band_info["color"],
            "label": band_info["label"]
        }
    
    segments["full"] = {
        "bands": full_bands,
        "start_times": [0],
        "points_per_segment": len(samples[::full_step])
    }
    
    return segments


def precompute_waveform_segments(samples: np.ndarray, sample_rate: int, duration: float):
    """Pre-compute waveform segments for different time windows"""
    segments = {}
    window_sizes = [1, 2, 3, 5, 10]  # seconds (max 10 sec)
    
    for window_size in window_sizes:
        if window_size > duration:
            continue
            
        # Calculate samples per segment
        samples_per_window = int(window_size * sample_rate)
        target_points_per_segment = min(2000, samples_per_window)
        step = max(1, samples_per_window // target_points_per_segment)
        
        # Calculate number of segments (with overlap for smooth scrolling)
        hop_seconds = window_size / 4  # 75% overlap
        hop_samples = int(hop_seconds * sample_rate)
        
        segment_list = []
        start_times = []
        
        for start in range(0, len(samples) - samples_per_window + 1, hop_samples):
            segment = samples[start:start + samples_per_window:step]
            segment_list.append(segment.tolist())
            start_times.append(start / sample_rate)
        
        segments[window_size] = {
            "segments": segment_list,
            "start_times": start_times,
            "points_per_segment": len(segment_list[0]) if segment_list else 0
        }
    
    # Also add "full" view
    full_step = max(1, len(samples) // 3000)
    segments["full"] = {
        "segments": [samples[::full_step].tolist()],
        "start_times": [0],
        "points_per_segment": len(samples[::full_step])
    }
    
    return segments


@app.get("/")
async def root():
    return {"message": "Audio Waveform API is running"}


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload audio file and pre-process ALL visualizations.
    Returns session_id to use with WebSocket.
    """
    allowed_extensions = ('.mp3', '.wav', '.ogg', '.m4a', '.flac')
    if not file.filename.lower().endswith(allowed_extensions):
        return JSONResponse(
            status_code=400,
            content={"error": f"Upload audio file ({', '.join(allowed_extensions)})"}
        )
    
    try:
        # Save temporarily
        contents = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Load audio
        samples, sample_rate = load_audio_file(tmp_path)
        os.unlink(tmp_path)
        
        duration = len(samples) / sample_rate
        session_id = str(uuid.uuid4())
        
        print(f"Processing audio: {file.filename} ({duration:.2f}s)")
        
        # PRE-COMPUTE EVERYTHING
        print("  Computing waveform...")
        waveform_data = precompute_waveform(samples, sample_rate)
        
        print("  Computing waveform segments...")
        waveform_segments = precompute_waveform_segments(samples, sample_rate, duration)
        
        print("  Computing FFT...")
        fft_data = precompute_fft(samples, sample_rate)
        
        print("  Computing spectrogram...")
        spectrogram_data = precompute_spectrogram(samples, sample_rate)
        
        print("  Computing power spectrum...")
        power_spectrum_data = precompute_power_spectrum(samples, sample_rate)
        
        print("  Computing frequency bands...")
        frequency_bands = precompute_frequency_bands(samples, sample_rate)
        
        print("  Computing frequency band segments...")
        frequency_band_segments = precompute_frequency_band_segments(samples, sample_rate, duration)
        
        # Store in cache
        audio_cache[session_id] = {
            "filename": file.filename,
            "duration": duration,
            "sample_rate": sample_rate,
            "waveform": waveform_data,
            "waveform_segments": waveform_segments,
            "fft": fft_data,
            "spectrogram": spectrogram_data,
            "power_spectrum": power_spectrum_data,
            "frequency_bands": frequency_bands,
            "frequency_band_segments": frequency_band_segments
        }
        
        print(f"  Done! Session: {session_id}")
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "duration": duration,
            "sample_rate": sample_rate,
            "available_windows": list(waveform_segments.keys())
        }
        
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing error: {str(e)}", "trace": traceback.format_exc()}
        )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time data streaming.
    
    Client sends:
    - {"type": "get_visualization", "viz_type": "waveform|fft|spectrogram|power_spectrum"}
    - {"type": "get_waveform_segment", "window_size": 5, "current_time": 10.5}
    
    Server sends back the pre-computed data.
    """
    await websocket.accept()
    
    if session_id not in audio_cache:
        await websocket.send_json({"error": "Session not found. Please upload a file first."})
        await websocket.close()
        return
    
    data = audio_cache[session_id]
    
    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")
            
            if msg_type == "get_visualization":
                viz_type = message.get("viz_type", "waveform")
                
                if viz_type == "waveform":
                    await websocket.send_json({
                        "type": "waveform",
                        "data": data["waveform"],
                        "duration": data["duration"]
                    })
                    
                elif viz_type == "fft":
                    if data["fft"]:
                        await websocket.send_json({"type": "fft", "data": data["fft"]})
                    else:
                        await websocket.send_json({"type": "error", "message": "Audio too short for FFT"})
                        
                elif viz_type == "spectrogram":
                    if data["spectrogram"]:
                        await websocket.send_json({"type": "spectrogram", "data": data["spectrogram"]})
                    else:
                        await websocket.send_json({"type": "error", "message": "Audio too short for spectrogram"})
                        
                elif viz_type == "power_spectrum":
                    if data["power_spectrum"]:
                        await websocket.send_json({"type": "power_spectrum", "data": data["power_spectrum"]})
                    else:
                        await websocket.send_json({"type": "error", "message": "Audio too short for power spectrum"})
            
            elif msg_type == "get_waveform_segment":
                window_size = message.get("window_size", 5)
                current_time = message.get("current_time", 0)
                duration = data["duration"]
                
                # Handle "full" as string
                if window_size == "full" or window_size >= duration:
                    window_key = "full"
                else:
                    window_key = window_size
                
                segments_data = data["waveform_segments"].get(window_key)
                
                if not segments_data:
                    # Fall back to full
                    segments_data = data["waveform_segments"].get("full")
                
                if segments_data:
                    # Find the best segment for current time
                    start_times = segments_data["start_times"]
                    segments = segments_data["segments"]
                    
                    # Find closest segment that contains current_time
                    best_idx = 0
                    for i, st in enumerate(start_times):
                        if st <= current_time:
                            best_idx = i
                        else:
                            break
                    
                    await websocket.send_json({
                        "type": "waveform_segment",
                        "segment": segments[best_idx],
                        "start_time": start_times[best_idx],
                        "window_size": window_key,
                        "duration": duration,
                        "max_amplitude": data["waveform"]["max_amplitude"]
                    })
                else:
                    await websocket.send_json({"type": "error", "message": "Segment not available"})
            
            elif msg_type == "get_frequency_bands":
                if data["frequency_bands"]:
                    await websocket.send_json({
                        "type": "frequency_bands",
                        "data": data["frequency_bands"],
                        "duration": data["duration"]
                    })
                else:
                    await websocket.send_json({"type": "error", "message": "Frequency bands not available"})
            
            elif msg_type == "get_band_segment":
                window_size = message.get("window_size", 5)
                current_time = message.get("current_time", 0)
                duration = data["duration"]
                
                print(f"get_band_segment: window_size={window_size}, current_time={current_time}")
                
                if window_size == "full" or window_size >= duration:
                    window_key = "full"
                else:
                    window_key = window_size
                
                print(f"  window_key={window_key}, available keys={list(data['frequency_band_segments'].keys())}")
                band_segments = data["frequency_band_segments"].get(window_key)
                
                if not band_segments:
                    band_segments = data["frequency_band_segments"].get("full")
                
                if band_segments:
                    start_times = band_segments["start_times"]
                    
                    # Find best segment index
                    best_idx = 0
                    for i, st in enumerate(start_times):
                        if st <= current_time:
                            best_idx = i
                        else:
                            break
                    
                    # Extract segments for this time
                    bands_data = {}
                    for band_name, band_info in band_segments["bands"].items():
                        bands_data[band_name] = {
                            "segment": band_info["segments"][best_idx],
                            "color": band_info["color"],
                            "label": band_info["label"]
                        }
                    
                    print(f"  Sending band_segment with {len(bands_data)} bands")
                    await websocket.send_json({
                        "type": "band_segment",
                        "bands": bands_data,
                        "start_time": start_times[best_idx],
                        "window_size": window_key,
                        "duration": duration,
                        "max_amplitude": data["waveform"]["max_amplitude"]
                    })
                else:
                    print(f"  ERROR: Band segments not available for window_key={window_key}")
                    await websocket.send_json({"type": "error", "message": "Band segments not available"})
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up session data"""
    if session_id in audio_cache:
        del audio_cache[session_id]
        return {"message": "Session deleted"}
    return {"message": "Session not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
