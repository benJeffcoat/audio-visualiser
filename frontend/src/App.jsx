import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

const VISUALIZATION_TYPES = [
  { id: 'waveform', label: 'Waveform', icon: '📈' },
  { id: 'fft', label: 'Frequency Spectrum', icon: '📊' },
  { id: 'spectrogram', label: 'Spectrogram', icon: '🌈' },
  { id: 'power_spectrum', label: 'Power Spectrum', icon: '⚡' },
]

function App() {
  const [audioData, setAudioData] = useState(null)
  const [analysisData, setAnalysisData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [windowSize, setWindowSize] = useState(5)
  const [vizType, setVizType] = useState('waveform')
  const [currentFile, setCurrentFile] = useState(null)
  
  const fileInputRef = useRef(null)
  const canvasRef = useRef(null)
  const audioRef = useRef(null)
  const animationRef = useRef(null)
  const rawAudioDataRef = useRef(null) // Store raw audio for client-side analysis

  // Fast FFT using Cooley-Tukey algorithm (O(n log n) instead of O(n²))
  const fft = useCallback((real, imag) => {
    const n = real.length
    if (n <= 1) return

    // Bit-reversal permutation
    for (let i = 0, j = 0; i < n; i++) {
      if (i < j) {
        [real[i], real[j]] = [real[j], real[i]]
        ;[imag[i], imag[j]] = [imag[j], imag[i]]
      }
      let k = n >> 1
      while (k <= j) {
        j -= k
        k >>= 1
      }
      j += k
    }

    // Cooley-Tukey FFT
    for (let len = 2; len <= n; len <<= 1) {
      const halfLen = len >> 1
      const angle = -2 * Math.PI / len
      const wReal = Math.cos(angle)
      const wImag = Math.sin(angle)

      for (let i = 0; i < n; i += len) {
        let curReal = 1, curImag = 0
        for (let j = 0; j < halfLen; j++) {
          const uReal = real[i + j]
          const uImag = imag[i + j]
          const tReal = curReal * real[i + j + halfLen] - curImag * imag[i + j + halfLen]
          const tImag = curReal * imag[i + j + halfLen] + curImag * real[i + j + halfLen]
          
          real[i + j] = uReal + tReal
          imag[i + j] = uImag + tImag
          real[i + j + halfLen] = uReal - tReal
          imag[i + j + halfLen] = uImag - tImag

          const newReal = curReal * wReal - curImag * wImag
          curImag = curReal * wImag + curImag * wReal
          curReal = newReal
        }
      }
    }
  }, [])

  // Client-side FFT analysis using fast algorithm
  const computeClientFFT = useCallback((rawData, sampleRate) => {
    // Use power of 2 for FFT efficiency
    const n_fft = Math.min(4096, Math.pow(2, Math.floor(Math.log2(rawData.length))))
    if (n_fft < 256) return null

    // Take middle section with Hanning window
    const start = Math.max(0, Math.floor(rawData.length / 2 - n_fft / 2))
    const real = new Float64Array(n_fft)
    const imag = new Float64Array(n_fft)

    for (let i = 0; i < n_fft; i++) {
      const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n_fft - 1)))
      real[i] = (rawData[start + i] || 0) * window
    }

    // Run FFT
    fft(real, imag)

    // Calculate magnitudes (only positive frequencies)
    const numBins = n_fft / 2
    const frequencies = []
    const magnitudes = []

    const maxFreqIdx = Math.min(numBins, Math.floor(20000 * n_fft / sampleRate))
    const targetPoints = 500
    const step = Math.max(1, Math.floor(maxFreqIdx / targetPoints))

    for (let i = 0; i < maxFreqIdx; i += step) {
      const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i])
      frequencies.push(i * sampleRate / n_fft)
      magnitudes.push(20 * Math.log10(mag + 1e-10))
    }

    return { frequencies, magnitudes }
  }, [fft])

  // Client-side Power Spectrum using fast FFT
  const computeClientPowerSpectrum = useCallback((rawData, sampleRate) => {
    const nperseg = Math.min(2048, Math.pow(2, Math.floor(Math.log2(rawData.length / 4))))
    if (nperseg < 256) return null

    const numBins = nperseg / 2
    const powerSum = new Float64Array(numBins)
    const hop = nperseg / 2
    const numSegments = Math.floor((rawData.length - nperseg) / hop) + 1

    // Hanning window
    const window = new Float64Array(nperseg)
    let windowPower = 0
    for (let i = 0; i < nperseg; i++) {
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nperseg - 1)))
      windowPower += window[i] * window[i]
    }

    // Process segments
    for (let seg = 0; seg < Math.min(numSegments, 20); seg++) { // Limit segments for speed
      const start = seg * hop
      const real = new Float64Array(nperseg)
      const imag = new Float64Array(nperseg)

      for (let i = 0; i < nperseg; i++) {
        real[i] = (rawData[start + i] || 0) * window[i]
      }

      fft(real, imag)

      for (let k = 0; k < numBins; k++) {
        powerSum[k] += (real[k] * real[k] + imag[k] * imag[k]) / (windowPower * sampleRate)
      }
    }

    // Average and convert to dB
    const actualSegments = Math.min(numSegments, 20)
    const frequencies = []
    const power = []
    const maxFreqIdx = Math.min(numBins, Math.floor(20000 * nperseg / sampleRate))
    const targetPoints = 500
    const step = Math.max(1, Math.floor(maxFreqIdx / targetPoints))

    for (let i = 0; i < maxFreqIdx; i += step) {
      frequencies.push(i * sampleRate / nperseg)
      power.push(10 * Math.log10(powerSum[i] / actualSegments + 1e-10))
    }

    return { frequencies, power }
  }, [fft])

  // Client-side Spectrogram using STFT
  const computeClientSpectrogram = useCallback((rawData, sampleRate) => {
    const nperseg = 1024
    const hop = 512
    const numBins = nperseg / 2

    const numFrames = Math.floor((rawData.length - nperseg) / hop) + 1
    if (numFrames < 2) return null

    // Limit frames for performance
    const maxFrames = 300
    const frameStep = Math.max(1, Math.floor(numFrames / maxFrames))
    const actualFrames = Math.ceil(numFrames / frameStep)

    // Hanning window
    const window = new Float64Array(nperseg)
    for (let i = 0; i < nperseg; i++) {
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nperseg - 1)))
    }

    const spectrogram = []
    const times = []
    let minDb = Infinity, maxDb = -Infinity

    // Limit frequency to ~10kHz for display
    const maxFreqBin = Math.min(numBins, Math.floor(10000 * nperseg / sampleRate))
    const freqStep = Math.max(1, Math.floor(maxFreqBin / 100))

    for (let frame = 0; frame < numFrames; frame += frameStep) {
      const start = frame * hop
      const real = new Float64Array(nperseg)
      const imag = new Float64Array(nperseg)

      for (let i = 0; i < nperseg; i++) {
        real[i] = (rawData[start + i] || 0) * window[i]
      }

      fft(real, imag)

      const frameMags = []
      for (let k = 0; k < maxFreqBin; k += freqStep) {
        const mag = Math.sqrt(real[k] * real[k] + imag[k] * imag[k])
        const db = 10 * Math.log10(mag + 1e-10)
        frameMags.push(db)
        if (db < minDb) minDb = db
        if (db > maxDb) maxDb = db
      }
      spectrogram.push(frameMags)
      times.push(start / sampleRate)
    }

    // Transpose for proper orientation (freq x time)
    const transposed = []
    const numFreqs = spectrogram[0].length
    for (let f = 0; f < numFreqs; f++) {
      transposed.push(spectrogram.map(frame => frame[f]))
    }

    const frequencies = []
    for (let k = 0; k < maxFreqBin; k += freqStep) {
      frequencies.push(k * sampleRate / nperseg)
    }

    return {
      spectrogram: transposed,
      times,
      frequencies,
      min_db: minDb,
      max_db: maxDb
    }
  }, [fft])

  // Perform analysis - ALL client-side now!
  const performAnalysis = useCallback(async (type) => {
    if (type === 'waveform') return
    if (!rawAudioDataRef.current) return

    setAnalysisLoading(true)
    setError(null)

    // Use setTimeout to avoid blocking UI
    setTimeout(() => {
      try {
        const { data, sampleRate } = rawAudioDataRef.current
        let result

        if (type === 'fft') {
          result = computeClientFFT(data, sampleRate)
        } else if (type === 'power_spectrum') {
          result = computeClientPowerSpectrum(data, sampleRate)
        } else if (type === 'spectrogram') {
          result = computeClientSpectrogram(data, sampleRate)
        }

        if (result) {
          setAnalysisData(result)
        } else {
          setError('Audio too short for analysis')
        }
      } catch (err) {
        setError(`Analysis error: ${err.message}`)
      } finally {
        setAnalysisLoading(false)
      }
    }, 50)
  }, [computeClientFFT, computeClientPowerSpectrum, computeClientSpectrogram])

  // Draw waveform
  const drawWaveform = useCallback((ctx, width, height, padding) => {
    if (!audioData?.waveform) return

    const { waveform, duration, maxAmplitude } = audioData
    const centerY = height / 2

    // Calculate visible time range
    const halfWindow = windowSize / 2
    let startTime = currentTime - halfWindow
    let endTime = currentTime + halfWindow

    if (startTime < 0) {
      startTime = 0
      endTime = Math.min(windowSize, duration)
    }
    if (endTime > duration) {
      endTime = duration
      startTime = Math.max(0, duration - windowSize)
    }

    const samplesPerSecond = waveform.length / duration
    const startSample = Math.floor(startTime * samplesPerSecond)
    const endSample = Math.ceil(endTime * samplesPerSecond)
    let visibleWaveform = waveform.slice(startSample, endSample)
    const visibleDuration = endTime - startTime

    // PERFORMANCE: Downsample visible waveform to max ~2000 points
    const maxRenderPoints = 2000
    if (visibleWaveform.length > maxRenderPoints) {
      const downsampleStep = Math.ceil(visibleWaveform.length / maxRenderPoints)
      const downsampled = []
      for (let i = 0; i < visibleWaveform.length; i += downsampleStep) {
        // Take max absolute value in each chunk for better peak representation
        let maxVal = visibleWaveform[i]
        let minVal = visibleWaveform[i]
        for (let j = i; j < Math.min(i + downsampleStep, visibleWaveform.length); j++) {
          if (visibleWaveform[j] > maxVal) maxVal = visibleWaveform[j]
          if (visibleWaveform[j] < minVal) minVal = visibleWaveform[j]
        }
        // Push the value with larger absolute magnitude
        downsampled.push(Math.abs(maxVal) > Math.abs(minVal) ? maxVal : minVal)
      }
      visibleWaveform = downsampled
    }

    // Draw grid
    ctx.strokeStyle = '#1a1a26'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = padding + ((height - padding * 2) / 4) * i
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }

    const timeStep = visibleDuration <= 5 ? 0.5 : visibleDuration <= 20 ? 1 : Math.ceil(visibleDuration / 10)
    const firstGridTime = Math.ceil(startTime / timeStep) * timeStep
    for (let t = firstGridTime; t <= endTime; t += timeStep) {
      const x = padding + ((t - startTime) / visibleDuration) * (width - padding * 2)
      ctx.beginPath()
      ctx.moveTo(x, padding)
      ctx.lineTo(x, height - padding)
      ctx.stroke()
    }

    // Center line
    ctx.strokeStyle = '#2a2a3a'
    ctx.beginPath()
    ctx.moveTo(padding, centerY)
    ctx.lineTo(width - padding, centerY)
    ctx.stroke()

    const drawWidth = width - padding * 2
    const drawHeight = (height - padding * 2) / 2

    // Draw waveform using FIXED global maxAmplitude
    const gradient = ctx.createLinearGradient(0, padding, 0, height - padding)
    gradient.addColorStop(0, '#818cf8')
    gradient.addColorStop(0.5, '#6366f1')
    gradient.addColorStop(1, '#818cf8')
    ctx.strokeStyle = gradient
    ctx.lineWidth = 1.5
    ctx.beginPath()

    visibleWaveform.forEach((sample, i) => {
      const x = padding + (i / visibleWaveform.length) * drawWidth
      const y = centerY - (sample / maxAmplitude) * drawHeight * 0.9
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Fill
    ctx.fillStyle = 'rgba(99, 102, 241, 0.1)'
    ctx.beginPath()
    ctx.moveTo(padding, centerY)
    visibleWaveform.forEach((sample, i) => {
      const x = padding + (i / visibleWaveform.length) * drawWidth
      const y = centerY - (sample / maxAmplitude) * drawHeight * 0.9
      ctx.lineTo(x, y)
    })
    ctx.lineTo(width - padding, centerY)
    ctx.closePath()
    ctx.fill()

    // Playhead
    if (currentTime >= startTime && currentTime <= endTime) {
      const playheadX = padding + ((currentTime - startTime) / visibleDuration) * drawWidth
      ctx.strokeStyle = '#10b981'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(playheadX, padding)
      ctx.lineTo(playheadX, height - padding)
      ctx.stroke()
      ctx.fillStyle = '#10b981'
      ctx.beginPath()
      ctx.moveTo(playheadX, padding - 8)
      ctx.lineTo(playheadX - 6, padding)
      ctx.lineTo(playheadX + 6, padding)
      ctx.closePath()
      ctx.fill()
    }

    // Labels - using FIXED maxAmplitude
    ctx.fillStyle = '#55556a'
    ctx.font = '11px Outfit'
    ctx.textAlign = 'right'
    ctx.fillText(maxAmplitude.toFixed(2), padding - 8, padding + 4)
    ctx.fillText('0', padding - 8, centerY + 4)
    ctx.fillText(`-${maxAmplitude.toFixed(2)}`, padding - 8, height - padding + 4)

    ctx.textAlign = 'center'
    for (let t = firstGridTime; t <= endTime; t += timeStep) {
      const x = padding + ((t - startTime) / visibleDuration) * (width - padding * 2)
      ctx.fillText(`${t.toFixed(1)}s`, x, height - padding + 20)
    }

    ctx.fillStyle = '#8888a0'
    ctx.font = '12px Outfit'
    ctx.save()
    ctx.translate(12, centerY)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Amplitude', 0, 0)
    ctx.restore()
    ctx.textAlign = 'center'
    ctx.fillText('Time (seconds)', width / 2, height - 8)
  }, [audioData, currentTime, windowSize])

  // Draw FFT / Frequency Spectrum
  const drawFFT = useCallback((ctx, width, height, padding) => {
    if (!analysisData?.frequencies || !analysisData?.magnitudes) return

    const { frequencies, magnitudes } = analysisData
    const drawWidth = width - padding * 2
    const drawHeight = height - padding * 2

    // Grid
    ctx.strokeStyle = '#1a1a26'
    ctx.lineWidth = 1
    for (let i = 0; i <= 5; i++) {
      const y = padding + (drawHeight / 5) * i
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }

    // Find range
    const minMag = Math.min(...magnitudes)
    const maxMag = Math.max(...magnitudes)
    const range = maxMag - minMag || 1

    // Draw spectrum
    const gradient = ctx.createLinearGradient(0, height - padding, 0, padding)
    gradient.addColorStop(0, '#6366f1')
    gradient.addColorStop(0.5, '#818cf8')
    gradient.addColorStop(1, '#c084fc')
    
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)

    frequencies.forEach((freq, i) => {
      const x = padding + (i / frequencies.length) * drawWidth
      const normalized = (magnitudes[i] - minMag) / range
      const y = height - padding - normalized * drawHeight * 0.9
      ctx.lineTo(x, y)
    })

    ctx.lineTo(width - padding, height - padding)
    ctx.closePath()
    ctx.fill()

    // Line on top
    ctx.strokeStyle = '#a5b4fc'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    frequencies.forEach((freq, i) => {
      const x = padding + (i / frequencies.length) * drawWidth
      const normalized = (magnitudes[i] - minMag) / range
      const y = height - padding - normalized * drawHeight * 0.9
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Labels
    ctx.fillStyle = '#55556a'
    ctx.font = '11px Outfit'
    ctx.textAlign = 'right'
    ctx.fillText(`${maxMag.toFixed(0)} dB`, padding - 8, padding + 4)
    ctx.fillText(`${((maxMag + minMag) / 2).toFixed(0)} dB`, padding - 8, height / 2)
    ctx.fillText(`${minMag.toFixed(0)} dB`, padding - 8, height - padding)

    ctx.textAlign = 'center'
    const maxFreq = frequencies[frequencies.length - 1]
    for (let f = 0; f <= maxFreq; f += 5000) {
      const x = padding + (f / maxFreq) * drawWidth
      ctx.fillText(`${(f / 1000).toFixed(0)}k`, x, height - padding + 20)
    }

    ctx.fillStyle = '#8888a0'
    ctx.font = '12px Outfit'
    ctx.save()
    ctx.translate(12, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Magnitude (dB)', 0, 0)
    ctx.restore()
    ctx.textAlign = 'center'
    ctx.fillText('Frequency (Hz)', width / 2, height - 8)
  }, [analysisData])

  // Draw Spectrogram
  const drawSpectrogram = useCallback((ctx, width, height, padding) => {
    if (!analysisData?.spectrogram) return

    const { spectrogram, times, frequencies, min_db, max_db } = analysisData
    const drawWidth = width - padding * 2
    const drawHeight = height - padding * 2

    const numFreqs = spectrogram.length
    const numTimes = spectrogram[0]?.length || 0
    
    const cellWidth = drawWidth / numTimes
    const cellHeight = drawHeight / numFreqs

    // Color mapping function (viridis-like)
    const getColor = (value) => {
      const t = Math.max(0, Math.min(1, (value - min_db) / (max_db - min_db)))
      const r = Math.floor(68 + t * 187)
      const g = Math.floor(1 + t * 150)
      const b = Math.floor(84 + t * 100)
      return `rgb(${r}, ${g}, ${b})`
    }

    // Draw spectrogram cells
    for (let f = 0; f < numFreqs; f++) {
      for (let t = 0; t < numTimes; t++) {
        const value = spectrogram[f][t]
        ctx.fillStyle = getColor(value)
        const x = padding + t * cellWidth
        const y = height - padding - (f + 1) * cellHeight
        ctx.fillRect(x, y, cellWidth + 1, cellHeight + 1)
      }
    }

    // Labels
    ctx.fillStyle = '#55556a'
    ctx.font = '11px Outfit'
    ctx.textAlign = 'right'
    
    const maxFreq = frequencies[frequencies.length - 1]
    for (let f = 0; f <= maxFreq; f += 5000) {
      const y = height - padding - (f / maxFreq) * drawHeight
      ctx.fillText(`${(f / 1000).toFixed(0)}k`, padding - 8, y + 4)
    }

    ctx.textAlign = 'center'
    const maxTime = times[times.length - 1]
    const timeStep = Math.ceil(maxTime / 8)
    for (let t = 0; t <= maxTime; t += timeStep) {
      const x = padding + (t / maxTime) * drawWidth
      ctx.fillText(`${t.toFixed(1)}s`, x, height - padding + 20)
    }

    ctx.fillStyle = '#8888a0'
    ctx.font = '12px Outfit'
    ctx.save()
    ctx.translate(12, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Frequency (Hz)', 0, 0)
    ctx.restore()
    ctx.textAlign = 'center'
    ctx.fillText('Time (seconds)', width / 2, height - 8)

    // Color bar
    const barWidth = 15
    const barX = width - padding + 10
    const barHeight = drawHeight
    const barGradient = ctx.createLinearGradient(0, height - padding, 0, padding)
    for (let i = 0; i <= 10; i++) {
      const t = i / 10
      barGradient.addColorStop(t, getColor(min_db + t * (max_db - min_db)))
    }
    ctx.fillStyle = barGradient
    ctx.fillRect(barX, padding, barWidth, barHeight)
    
    ctx.fillStyle = '#55556a'
    ctx.font = '10px Outfit'
    ctx.textAlign = 'left'
    ctx.fillText(`${max_db.toFixed(0)}`, barX + barWidth + 4, padding + 8)
    ctx.fillText(`${min_db.toFixed(0)}`, barX + barWidth + 4, height - padding)
    ctx.fillText('dB', barX + barWidth + 4, height / 2)
  }, [analysisData])

  // Draw Power Spectrum
  const drawPowerSpectrum = useCallback((ctx, width, height, padding) => {
    if (!analysisData?.frequencies || !analysisData?.power) return

    const { frequencies, power } = analysisData
    const drawWidth = width - padding * 2
    const drawHeight = height - padding * 2

    // Grid
    ctx.strokeStyle = '#1a1a26'
    ctx.lineWidth = 1
    for (let i = 0; i <= 5; i++) {
      const y = padding + (drawHeight / 5) * i
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }

    const minPow = Math.min(...power)
    const maxPow = Math.max(...power)
    const range = maxPow - minPow || 1

    // Fill under curve
    const gradient = ctx.createLinearGradient(0, height - padding, 0, padding)
    gradient.addColorStop(0, 'rgba(16, 185, 129, 0.3)')
    gradient.addColorStop(1, 'rgba(16, 185, 129, 0.05)')
    
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)

    frequencies.forEach((freq, i) => {
      const x = padding + (i / frequencies.length) * drawWidth
      const normalized = (power[i] - minPow) / range
      const y = height - padding - normalized * drawHeight * 0.9
      ctx.lineTo(x, y)
    })

    ctx.lineTo(width - padding, height - padding)
    ctx.closePath()
    ctx.fill()

    // Line
    ctx.strokeStyle = '#10b981'
    ctx.lineWidth = 2
    ctx.beginPath()
    frequencies.forEach((freq, i) => {
      const x = padding + (i / frequencies.length) * drawWidth
      const normalized = (power[i] - minPow) / range
      const y = height - padding - normalized * drawHeight * 0.9
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()

    // Labels
    ctx.fillStyle = '#55556a'
    ctx.font = '11px Outfit'
    ctx.textAlign = 'right'
    ctx.fillText(`${maxPow.toFixed(0)} dB`, padding - 8, padding + 4)
    ctx.fillText(`${minPow.toFixed(0)} dB`, padding - 8, height - padding)

    ctx.textAlign = 'center'
    const maxFreq = frequencies[frequencies.length - 1]
    for (let f = 0; f <= maxFreq; f += 5000) {
      const x = padding + (f / maxFreq) * drawWidth
      ctx.fillText(`${(f / 1000).toFixed(0)}k`, x, height - padding + 20)
    }

    ctx.fillStyle = '#8888a0'
    ctx.font = '12px Outfit'
    ctx.save()
    ctx.translate(12, height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText('Power (dB/Hz)', 0, 0)
    ctx.restore()
    ctx.textAlign = 'center'
    ctx.fillText('Frequency (Hz)', width / 2, height - 8)
  }, [analysisData])

  // Main draw function
  const draw = useCallback(() => {
    if (!canvasRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const container = canvas.parentElement
    canvas.width = container.clientWidth
    canvas.height = container.clientHeight

    const width = canvas.width
    const height = canvas.height
    const padding = vizType === 'spectrogram' ? 50 : 40

    // Clear
    ctx.fillStyle = '#0a0a0f'
    ctx.fillRect(0, 0, width, height)

    if (vizType === 'waveform') {
      drawWaveform(ctx, width, height, padding)
    } else if (vizType === 'fft') {
      drawFFT(ctx, width, height, padding)
    } else if (vizType === 'spectrogram') {
      drawSpectrogram(ctx, width, height, padding)
    } else if (vizType === 'power_spectrum') {
      drawPowerSpectrum(ctx, width, height, padding)
    }
  }, [vizType, drawWaveform, drawFFT, drawSpectrogram, drawPowerSpectrum])

  // Handle viz type change
  useEffect(() => {
    if (audioData && vizType !== 'waveform') {
      performAnalysis(vizType)
    }
  }, [vizType, audioData, performAnalysis])

  // Animation loop
  const updatePlayback = useCallback(() => {
    if (audioRef.current && !audioRef.current.paused) {
      setCurrentTime(audioRef.current.currentTime)
      animationRef.current = requestAnimationFrame(updatePlayback)
    }
  }, [])

  // Resize handler
  useEffect(() => {
    const handleResize = () => draw()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [draw])

  // Draw when data changes
  useEffect(() => {
    if (audioData || analysisData) {
      setTimeout(draw, 50)
    }
  }, [audioData, analysisData, draw, currentTime])

  // Cleanup
  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [])

  // File upload handler
  const handleFileUpload = async (file) => {
    if (!file) return

    setLoading(true)
    setError(null)
    setCurrentFile(file)
    setVizType('waveform')
    setAnalysisData(null)

    try {
      const arrayBuffer = await file.arrayBuffer()
      const audioContext = new (window.AudioContext || window.webkitAudioContext)()
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)
      const channelData = audioBuffer.getChannelData(0)
      
      // Store raw audio data for client-side analysis
      rawAudioDataRef.current = {
        data: new Float32Array(channelData),
        sampleRate: audioBuffer.sampleRate
      }
      
      const targetPoints = 10000
      const step = Math.max(1, Math.floor(channelData.length / targetPoints))
      const waveform = []
      let globalMaxAmplitude = 0
      
      for (let i = 0; i < channelData.length; i += step) {
        const sample = channelData[i]
        waveform.push(sample)
        const abs = Math.abs(sample)
        if (abs > globalMaxAmplitude) globalMaxAmplitude = abs
      }
      
      // Ensure minimum scale
      globalMaxAmplitude = Math.max(globalMaxAmplitude, 0.01) * 1.05

      const audioUrl = URL.createObjectURL(file)
      
      setAudioData({
        filename: file.name,
        duration: audioBuffer.duration,
        sampleRate: audioBuffer.sampleRate,
        channels: audioBuffer.numberOfChannels,
        waveform,
        points: waveform.length,
        audioUrl,
        maxAmplitude: globalMaxAmplitude
      })
      
      setCurrentTime(0)
      setIsPlaying(false)
      audioContext.close()
    } catch (err) {
      setError('Failed to process audio: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const togglePlayback = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    } else {
      audioRef.current.play()
      animationRef.current = requestAnimationFrame(updatePlayback)
    }
    setIsPlaying(!isPlaying)
  }

  const handleSeek = (e) => {
    if (!audioRef.current || !audioData) return
    const rect = e.currentTarget.getBoundingClientRect()
    const percentage = (e.clientX - rect.left) / rect.width
    const newTime = percentage * audioData.duration
    audioRef.current.currentTime = newTime
    setCurrentTime(newTime)
  }

  const handleAudioEnded = () => {
    setIsPlaying(false)
    setCurrentTime(0)
    if (animationRef.current) cancelAnimationFrame(animationRef.current)
  }

  const handleDragOver = (e) => { e.preventDefault(); setDragOver(true) }
  const handleDragLeave = (e) => { e.preventDefault(); setDragOver(false) }
  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0])
  }
  const handleInputChange = (e) => {
    if (e.target.files[0]) handleFileUpload(e.target.files[0])
  }

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const clearAudio = () => {
    if (audioRef.current) audioRef.current.pause()
    if (animationRef.current) cancelAnimationFrame(animationRef.current)
    if (audioData?.audioUrl) URL.revokeObjectURL(audioData.audioUrl)
    rawAudioDataRef.current = null
    setAudioData(null)
    setAnalysisData(null)
    setCurrentFile(null)
    setError(null)
    setIsPlaying(false)
    setCurrentTime(0)
    setVizType('waveform')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const vizLabel = VISUALIZATION_TYPES.find(v => v.id === vizType)?.label || 'Visualization'

  return (
    <div className="app-container">
      {audioData && (
        <audio
          ref={audioRef}
          src={audioData.audioUrl}
          onEnded={handleAudioEnded}
          onTimeUpdate={() => setCurrentTime(audioRef.current?.currentTime || 0)}
        />
      )}

      <header className="header">
        <div className="header-logo">
          <div className="logo-icon">🎵</div>
          <span className="logo-text">WaveForm</span>
        </div>
        <div className="header-actions">
          <div className="header-status">
            <span className="status-dot"></span>
            <span>Ready</span>
          </div>
        </div>
      </header>

      <aside className="sidebar">
        <div className="sidebar-section">
          <div className="sidebar-title">Menu</div>
          <div className="sidebar-item active">
            <span className="sidebar-item-icon">📊</span>
            <span>Visualizer</span>
          </div>
          <div className="sidebar-item">
            <span className="sidebar-item-icon">📁</span>
            <span>Library</span>
          </div>
          <div className="sidebar-item">
            <span className="sidebar-item-icon">⚙️</span>
            <span>Settings</span>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-title">Info</div>
          <div className="sidebar-item">
            <span className="sidebar-item-icon">❓</span>
            <span>Help</span>
          </div>
        </div>

        <div
          className={`upload-section ${dragOver ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="upload-content">
            <div className="upload-icon">⬆️</div>
            <div className="upload-text">
              Drop audio file here or{' '}
              <span onClick={() => fileInputRef.current?.click()}>browse</span>
            </div>
            <div className="upload-formats">MP3, WAV, OGG, M4A</div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            className="upload-input"
            accept=".mp3,.wav,.ogg,.m4a,audio/*"
            onChange={handleInputChange}
          />
        </div>
      </aside>

      <main className="main-content">
        {audioData && (
          <div className="file-info-card">
            <div className="file-details">
              <div className="file-icon">🎶</div>
              <div>
                <div className="file-name">{audioData.filename}</div>
                <div className="file-meta">
                  {formatDuration(audioData.duration)} • {audioData.sampleRate} Hz
                </div>
              </div>
            </div>
            <div className="file-actions">
              <button className="btn btn-ghost" onClick={clearAudio}>Clear</button>
            </div>
          </div>
        )}

        {audioData && (
          <div className="player-controls">
            <button className="play-btn" onClick={togglePlayback}>
              {isPlaying ? '⏸️' : '▶️'}
            </button>
            
            <div className="time-display">
              <span className="current-time">{formatDuration(currentTime)}</span>
              <span className="time-separator">/</span>
              <span className="total-time">{formatDuration(audioData.duration)}</span>
            </div>

            <div className="progress-container" onClick={handleSeek}>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${(currentTime / audioData.duration) * 100}%` }} />
                <div className="progress-handle" style={{ left: `${(currentTime / audioData.duration) * 100}%` }} />
              </div>
            </div>

            {vizType === 'waveform' && (
              <div className="window-control">
                <label className="window-label">Window:</label>
                <select 
                  className="window-select"
                  value={windowSize}
                  onChange={(e) => setWindowSize(Number(e.target.value))}
                >
                  <option value={1}>1 sec</option>
                  <option value={2}>2 sec</option>
                  <option value={3}>3 sec</option>
                  <option value={5}>5 sec</option>
                  <option value={10}>10 sec</option>
                  <option value={30}>30 sec</option>
                  <option value={60}>1 min</option>
                  <option value={audioData.duration}>Full</option>
                </select>
              </div>
            )}

            <div className="viz-control">
              <label className="window-label">View:</label>
              <select 
                className="window-select viz-select"
                value={vizType}
                onChange={(e) => setVizType(e.target.value)}
              >
                {VISUALIZATION_TYPES.map(v => (
                  <option key={v.id} value={v.id}>{v.icon} {v.label}</option>
                ))}
              </select>
            </div>
          </div>
        )}

        <div className="waveform-container">
          <div className="waveform-header">
            <h2 className="waveform-title">{vizLabel}</h2>
            {audioData && (
              <div className="waveform-info">
                {vizType === 'waveform' && (
                  <div className="info-badge">
                    Window: <span className="info-badge-value">{windowSize}s</span>
                  </div>
                )}
                <div className="info-badge">
                  Sample Rate: <span className="info-badge-value">{audioData.sampleRate} Hz</span>
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className="error-state">
              <span>⚠️</span>
              <span>{error}</span>
            </div>
          )}

          {(loading || analysisLoading) ? (
            <div className="loading-state">
              <div className="spinner"></div>
              <div className="loading-text">
                {analysisLoading ? 'Analyzing audio...' : 'Processing audio file...'}
              </div>
            </div>
          ) : audioData ? (
            <div className="canvas-container">
              <canvas ref={canvasRef} className="waveform-canvas"></canvas>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">🎧</div>
              <div className="empty-title">No Audio Loaded</div>
              <div className="empty-text">
                Upload an audio file using the sidebar to visualize its waveform
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
