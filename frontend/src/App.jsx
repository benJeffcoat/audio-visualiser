import { useState, useRef, useEffect, useCallback } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000'

const VISUALIZATION_TYPES = [
  { id: 'waveform', label: 'Waveform', icon: '📈' },
  { id: 'fft', label: 'Frequency Spectrum', icon: '📊' },
  { id: 'spectrogram', label: 'Spectrogram', icon: '🌈' },
  { id: 'power_spectrum', label: 'Power Spectrum', icon: '⚡' },
]

const FREQUENCY_BANDS = [
  { id: 'sub_bass', label: 'Sub Bass', color: '#ef4444', shortLabel: 'Sub' },
  { id: 'bass', label: 'Bass', color: '#f97316', shortLabel: 'Bass' },
  { id: 'low_mid', label: 'Low Mid', color: '#eab308', shortLabel: 'Lo-Mid' },
  { id: 'mid', label: 'Mid', color: '#22c55e', shortLabel: 'Mid' },
  { id: 'high_mid', label: 'High Mid', color: '#3b82f6', shortLabel: 'Hi-Mid' },
  { id: 'high', label: 'High', color: '#a855f7', shortLabel: 'High' },
]

function App() {
  // Audio state
  const [sessionId, setSessionId] = useState(null)
  const [audioInfo, setAudioInfo] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)
  
  // Visualization data (received from backend)
  const [vizData, setVizData] = useState(null)
  const [waveformSegment, setWaveformSegment] = useState(null)
  const [bandSegment, setBandSegment] = useState(null)
  
  // Frequency band mode
  const [showBands, setShowBands] = useState(false)
  const [visibleBands, setVisibleBands] = useState({
    sub_bass: true,
    bass: true,
    low_mid: true,
    mid: true,
    high_mid: true,
    high: true
  })
  
  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [windowSize, setWindowSize] = useState(5)
  const [vizType, setVizType] = useState('waveform')
  const [wsConnected, setWsConnected] = useState(false)
  
  // Refs
  const fileInputRef = useRef(null)
  const canvasRef = useRef(null)
  const audioRef = useRef(null)
  const wsRef = useRef(null)
  const animationRef = useRef(null)
  const currentTimeRef = useRef(0) // For smooth animation

  // Connect WebSocket
  const connectWebSocket = useCallback((sid) => {
    if (wsRef.current) {
      wsRef.current.close()
    }
    
    const ws = new WebSocket(`${WS_URL}/ws/${sid}`)
    
    ws.onopen = () => {
      console.log('WebSocket connected')
      setWsConnected(true)
      // Request initial waveform data
      ws.send(JSON.stringify({ type: 'get_visualization', viz_type: 'waveform' }))
    }
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'waveform') {
        setVizData({ type: 'waveform', ...data.data, duration: data.duration })
      } else if (data.type === 'fft') {
        setVizData({ type: 'fft', ...data.data })
      } else if (data.type === 'spectrogram') {
        setVizData({ type: 'spectrogram', ...data.data })
      } else if (data.type === 'power_spectrum') {
        setVizData({ type: 'power_spectrum', ...data.data })
      } else if (data.type === 'waveform_segment') {
        setWaveformSegment(data)
      } else if (data.type === 'band_segment') {
        setBandSegment(data)
      } else if (data.type === 'error') {
        console.error('WebSocket error:', data.message)
        setError(data.message)
      }
    }
    
    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setWsConnected(false)
    }
    
    ws.onerror = (err) => {
      console.error('WebSocket error:', err)
      setError('WebSocket connection failed')
    }
    
    wsRef.current = ws
  }, [])

  // Request visualization data
  const requestVisualization = useCallback((type) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'get_visualization', viz_type: type }))
    }
  }, [])

  // Request waveform segment for current time
  const requestWaveformSegment = useCallback((time, winSize) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'get_waveform_segment',
        window_size: winSize,
        current_time: time
      }))
    }
  }, [])

  // Request frequency band segment
  const requestBandSegment = useCallback((time, winSize) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'get_band_segment',
        window_size: winSize,
        current_time: time
      }))
    }
  }, [])

  // Request band segment immediately when showBands is enabled
  useEffect(() => {
    if (showBands && wsConnected && audioInfo) {
      const winSize = windowSize >= audioInfo.duration ? 'full' : windowSize
      requestBandSegment(currentTime, winSize)
    }
  }, [showBands, wsConnected, audioInfo, windowSize, currentTime, requestBandSegment])

  // Handle viz type change
  useEffect(() => {
    if (wsConnected && vizType) {
      requestVisualization(vizType)
    }
  }, [vizType, wsConnected, requestVisualization])

  // Request waveform segments only when needed (not every frame)
  const lastSegmentRequestRef = useRef({ time: 0, windowSize: null, showBands: false })
  
  useEffect(() => {
    if (vizType === 'waveform' && wsConnected && audioInfo) {
      const winSize = windowSize >= audioInfo.duration ? 'full' : windowSize
      
      // For 'full' view, only request once
      if (winSize === 'full') {
        if (lastSegmentRequestRef.current.windowSize !== 'full' || lastSegmentRequestRef.current.showBands !== showBands) {
          requestWaveformSegment(0, 'full')
          if (showBands) requestBandSegment(0, 'full')
          lastSegmentRequestRef.current = { time: 0, windowSize: 'full', showBands }
        }
        return
      }
      
      // For windowed view, only request new segment when approaching boundaries
      const segmentStart = waveformSegment?.start_time ?? 0
      const segmentEnd = segmentStart + (typeof winSize === 'number' ? winSize : 5)
      
      // Request new segment if:
      // 1. Window size changed
      // 2. Current time is past 75% of segment or before start
      // 3. showBands toggled
      const needsNewSegment = 
        lastSegmentRequestRef.current.windowSize !== winSize ||
        lastSegmentRequestRef.current.showBands !== showBands ||
        !waveformSegment ||
        currentTime < segmentStart ||
        currentTime > segmentStart + (segmentEnd - segmentStart) * 0.75
      
      if (needsNewSegment) {
        requestWaveformSegment(currentTime, winSize)
        if (showBands) requestBandSegment(currentTime, winSize)
        lastSegmentRequestRef.current = { time: currentTime, windowSize: winSize, showBands }
      }
    }
  }, [currentTime, windowSize, vizType, wsConnected, audioInfo, requestWaveformSegment, requestBandSegment, waveformSegment, showBands])

  // ============ DRAWING FUNCTIONS (visualization only) ============
  
  const drawWaveform = useCallback((ctx, width, height, padding) => {
    if (!waveformSegment) return
    
    const { segment, max_amplitude, start_time, window_size, duration } = waveformSegment
    const centerY = height / 2
    const drawWidth = width - padding * 2
    const drawHeight = (height - padding * 2) / 2
    
    // Duration of this segment
    const totalDuration = duration || audioInfo?.duration || 1
    const segmentDuration = window_size === 'full' ? totalDuration : window_size
    const endTime = start_time + segmentDuration
    
    // Grid
    ctx.strokeStyle = '#1a1a26'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = padding + ((height - padding * 2) / 4) * i
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()
    }
    
    // Time grid
    const timeStep = segmentDuration <= 5 ? 0.5 : segmentDuration <= 20 ? 1 : Math.ceil(segmentDuration / 10)
    const firstGridTime = Math.ceil(start_time / timeStep) * timeStep
    
    for (let t = firstGridTime; t <= endTime; t += timeStep) {
      const x = padding + ((t - start_time) / segmentDuration) * drawWidth
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
    
    // Draw frequency bands if enabled
    if (showBands && bandSegment && bandSegment.bands) {
      // Draw each visible band - normalize each band to make them visible
      const bandOrder = ['sub_bass', 'bass', 'low_mid', 'mid', 'high_mid', 'high']
      
      // First pass: calculate max amplitude for each visible band
      const bandMaxAmplitudes = {}
      bandOrder.forEach(bandName => {
        if (!visibleBands[bandName]) return
        const band = bandSegment.bands[bandName]
        if (!band?.segment) return
        const bandMax = Math.max(...band.segment.map(Math.abs))
        bandMaxAmplitudes[bandName] = bandMax > 0 ? bandMax : 0.001
      })
      
      // Draw each band
      bandOrder.forEach(bandName => {
        if (!visibleBands[bandName]) return
        const band = bandSegment.bands[bandName]
        if (!band?.segment) return
        
        const bandData = band.segment
        const color = band.color
        const bandMax = bandMaxAmplitudes[bandName]
        
        // Draw band waveform
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.globalAlpha = 0.85
        ctx.beginPath()
        
        bandData.forEach((sample, i) => {
          const x = padding + (i / bandData.length) * drawWidth
          // Normalize each band to its own max amplitude so it fills the display
          const y = centerY - (sample / bandMax) * drawHeight * 0.8
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        })
        ctx.stroke()
        ctx.globalAlpha = 1
      })
    } else {
      // Original single waveform
      const gradient = ctx.createLinearGradient(0, padding, 0, height - padding)
      gradient.addColorStop(0, '#818cf8')
      gradient.addColorStop(0.5, '#6366f1')
      gradient.addColorStop(1, '#818cf8')
      ctx.strokeStyle = gradient
      ctx.lineWidth = 1.5
      ctx.beginPath()
      
      segment.forEach((sample, i) => {
        const x = padding + (i / segment.length) * drawWidth
        const y = centerY - (sample / max_amplitude) * drawHeight * 0.9
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.stroke()
      
      // Fill
      ctx.fillStyle = 'rgba(99, 102, 241, 0.1)'
      ctx.beginPath()
      ctx.moveTo(padding, centerY)
      segment.forEach((sample, i) => {
        const x = padding + (i / segment.length) * drawWidth
        const y = centerY - (sample / max_amplitude) * drawHeight * 0.9
        ctx.lineTo(x, y)
      })
      ctx.lineTo(width - padding, centerY)
      ctx.closePath()
      ctx.fill()
    }
    
    // Playhead - smooth movement through the waveform
    // Use ref value for smoother animation when playing
    const playTime = currentTimeRef.current || currentTime
    if (playTime >= start_time && playTime <= endTime) {
      const playheadX = padding + ((playTime - start_time) / segmentDuration) * drawWidth
      
      // Draw playhead line
      ctx.strokeStyle = '#10b981'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(playheadX, padding)
      ctx.lineTo(playheadX, height - padding)
      ctx.stroke()
      
      // Playhead triangle
      ctx.fillStyle = '#10b981'
      ctx.beginPath()
      ctx.moveTo(playheadX, padding - 8)
      ctx.lineTo(playheadX - 6, padding)
      ctx.lineTo(playheadX + 6, padding)
      ctx.closePath()
      ctx.fill()
    }
    
    // Labels
    ctx.fillStyle = '#55556a'
    ctx.font = '11px Outfit'
    ctx.textAlign = 'right'
    ctx.fillText(max_amplitude.toFixed(2), padding - 8, padding + 4)
    ctx.fillText('0', padding - 8, centerY + 4)
    ctx.fillText(`-${max_amplitude.toFixed(2)}`, padding - 8, height - padding + 4)
    
    ctx.textAlign = 'center'
    for (let t = firstGridTime; t <= endTime; t += timeStep) {
      const x = padding + ((t - start_time) / segmentDuration) * drawWidth
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
  }, [waveformSegment, currentTime, audioInfo, showBands, bandSegment, visibleBands])

  const drawFFT = useCallback((ctx, width, height, padding) => {
    if (!vizData || vizData.type !== 'fft') return
    
    const { frequencies, magnitudes } = vizData
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
    
    const minMag = Math.min(...magnitudes)
    const maxMag = Math.max(...magnitudes)
    const range = maxMag - minMag || 1
    
    // Spectrum fill
    const gradient = ctx.createLinearGradient(0, height - padding, 0, padding)
    gradient.addColorStop(0, '#6366f1')
    gradient.addColorStop(0.5, '#818cf8')
    gradient.addColorStop(1, '#c084fc')
    
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)
    frequencies.forEach((_, i) => {
      const x = padding + (i / frequencies.length) * drawWidth
      const normalized = (magnitudes[i] - minMag) / range
      const y = height - padding - normalized * drawHeight * 0.9
      ctx.lineTo(x, y)
    })
    ctx.lineTo(width - padding, height - padding)
    ctx.closePath()
    ctx.fill()
    
    // Line
    ctx.strokeStyle = '#a5b4fc'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    frequencies.forEach((_, i) => {
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
  }, [vizData])

  const drawSpectrogram = useCallback((ctx, width, height, padding) => {
    if (!vizData || vizData.type !== 'spectrogram') return
    
    const { spectrogram, times, frequencies, min_db, max_db } = vizData
    const drawWidth = width - padding * 2
    const drawHeight = height - padding * 2
    
    const numFreqs = spectrogram.length
    const numTimes = spectrogram[0]?.length || 0
    const cellWidth = drawWidth / numTimes
    const cellHeight = drawHeight / numFreqs
    
    const getColor = (value) => {
      const t = Math.max(0, Math.min(1, (value - min_db) / (max_db - min_db)))
      return `rgb(${Math.floor(68 + t * 187)}, ${Math.floor(1 + t * 150)}, ${Math.floor(84 + t * 100)})`
    }
    
    for (let f = 0; f < numFreqs; f++) {
      for (let t = 0; t < numTimes; t++) {
        ctx.fillStyle = getColor(spectrogram[f][t])
        ctx.fillRect(padding + t * cellWidth, height - padding - (f + 1) * cellHeight, cellWidth + 1, cellHeight + 1)
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
    const barWidth = 15, barX = width - padding + 10
    const barGradient = ctx.createLinearGradient(0, height - padding, 0, padding)
    for (let i = 0; i <= 10; i++) {
      barGradient.addColorStop(i / 10, getColor(min_db + (i / 10) * (max_db - min_db)))
    }
    ctx.fillStyle = barGradient
    ctx.fillRect(barX, padding, barWidth, drawHeight)
    
    ctx.fillStyle = '#55556a'
    ctx.font = '10px Outfit'
    ctx.textAlign = 'left'
    ctx.fillText(`${max_db.toFixed(0)}`, barX + barWidth + 4, padding + 8)
    ctx.fillText(`${min_db.toFixed(0)}`, barX + barWidth + 4, height - padding)
    ctx.fillText('dB', barX + barWidth + 4, height / 2)
  }, [vizData])

  const drawPowerSpectrum = useCallback((ctx, width, height, padding) => {
    if (!vizData || vizData.type !== 'power_spectrum') return
    
    const { frequencies, power } = vizData
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
    
    // Fill
    const gradient = ctx.createLinearGradient(0, height - padding, 0, padding)
    gradient.addColorStop(0, 'rgba(16, 185, 129, 0.3)')
    gradient.addColorStop(1, 'rgba(16, 185, 129, 0.05)')
    
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.moveTo(padding, height - padding)
    frequencies.forEach((_, i) => {
      const x = padding + (i / frequencies.length) * drawWidth
      const y = height - padding - ((power[i] - minPow) / range) * drawHeight * 0.9
      ctx.lineTo(x, y)
    })
    ctx.lineTo(width - padding, height - padding)
    ctx.closePath()
    ctx.fill()
    
    // Line
    ctx.strokeStyle = '#10b981'
    ctx.lineWidth = 2
    ctx.beginPath()
    frequencies.forEach((_, i) => {
      const x = padding + (i / frequencies.length) * drawWidth
      const y = height - padding - ((power[i] - minPow) / range) * drawHeight * 0.9
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
  }, [vizData])

  // Main draw
  const draw = useCallback(() => {
    if (!canvasRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const container = canvas.parentElement
    canvas.width = container.clientWidth
    canvas.height = container.clientHeight
    
    const { width, height } = canvas
    const padding = vizType === 'spectrogram' ? 50 : 40
    
    ctx.fillStyle = '#0a0a0f'
    ctx.fillRect(0, 0, width, height)
    
    if (vizType === 'waveform') drawWaveform(ctx, width, height, padding)
    else if (vizType === 'fft') drawFFT(ctx, width, height, padding)
    else if (vizType === 'spectrogram') drawSpectrogram(ctx, width, height, padding)
    else if (vizType === 'power_spectrum') drawPowerSpectrum(ctx, width, height, padding)
  }, [vizType, drawWaveform, drawFFT, drawSpectrogram, drawPowerSpectrum])

  // Redraw on data change
  useEffect(() => {
    if (vizData || waveformSegment || bandSegment) {
      requestAnimationFrame(draw)
    }
  }, [vizData, waveformSegment, bandSegment, draw, currentTime])

  // Resize handler
  useEffect(() => {
    const handleResize = () => draw()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [draw])

  // Animation loop for playback - smooth drawing using ref
  const updatePlayback = useCallback(() => {
    if (audioRef.current && !audioRef.current.paused) {
      const time = audioRef.current.currentTime
      currentTimeRef.current = time
      
      // Update state less frequently (every 100ms) for UI display
      // But draw every frame for smooth playhead
      setCurrentTime(time)
      
      // Request next frame
      animationRef.current = requestAnimationFrame(updatePlayback)
    }
  }, [])

  // Cleanup
  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [])

  // ============ FILE UPLOAD ============
  
  const handleFileUpload = async (file) => {
    if (!file) return
    
    setLoading(true)
    setError(null)
    setVizData(null)
    setWaveformSegment(null)
    setBandSegment(null)
    setShowBands(false)
    setVizType('waveform')
    
    try {
      // Upload to backend for processing
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      })
      
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || 'Upload failed')
      }
      
      // Store session and audio info
      setSessionId(data.session_id)
      setAudioInfo({
        filename: data.filename,
        duration: data.duration,
        sampleRate: data.sample_rate,
        availableWindows: data.available_windows
      })
      
      // Create audio URL for playback
      const url = URL.createObjectURL(file)
      setAudioUrl(url)
      
      // Connect WebSocket
      connectWebSocket(data.session_id)
      
      setCurrentTime(0)
      setIsPlaying(false)
      
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // ============ PLAYBACK CONTROLS ============
  
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
    if (!audioRef.current || !audioInfo) return
    const rect = e.currentTarget.getBoundingClientRect()
    const percentage = (e.clientX - rect.left) / rect.width
    const newTime = percentage * audioInfo.duration
    audioRef.current.currentTime = newTime
    currentTimeRef.current = newTime
    setCurrentTime(newTime)
  }

  const handleAudioEnded = () => {
    setIsPlaying(false)
    currentTimeRef.current = 0
    setCurrentTime(0)
    if (animationRef.current) cancelAnimationFrame(animationRef.current)
  }

  const clearAudio = async () => {
    if (audioRef.current) audioRef.current.pause()
    if (animationRef.current) cancelAnimationFrame(animationRef.current)
    if (wsRef.current) wsRef.current.close()
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    
    // Clean up server session
    if (sessionId) {
      try {
        await fetch(`${API_URL}/session/${sessionId}`, { method: 'DELETE' })
      } catch (e) { /* ignore */ }
    }
    
    setSessionId(null)
    setAudioInfo(null)
    setAudioUrl(null)
    setVizData(null)
    setWaveformSegment(null)
    setBandSegment(null)
    setShowBands(false)
    setError(null)
    setIsPlaying(false)
    setCurrentTime(0)
    setVizType('waveform')
    setWsConnected(false)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // ============ UI HANDLERS ============
  
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

  const vizLabel = VISUALIZATION_TYPES.find(v => v.id === vizType)?.label || 'Visualization'

  return (
    <div className="app-container">
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
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
          <div className={`header-status ${wsConnected ? 'connected' : ''}`}>
            <span className="status-dot"></span>
            <span>{wsConnected ? 'Connected' : 'Disconnected'}</span>
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
        {audioInfo && (
          <div className="file-info-card">
            <div className="file-details">
              <div className="file-icon">🎶</div>
              <div>
                <div className="file-name">{audioInfo.filename}</div>
                <div className="file-meta">
                  {formatDuration(audioInfo.duration)} • {audioInfo.sampleRate} Hz
                </div>
              </div>
            </div>
            <div className="file-actions">
              <button className="btn btn-ghost" onClick={clearAudio}>Clear</button>
            </div>
          </div>
        )}

        {audioInfo && (
          <div className="player-controls">
            <button className="play-btn" onClick={togglePlayback}>
              {isPlaying ? '⏸️' : '▶️'}
            </button>
            
            <div className="time-display">
              <span className="current-time">{formatDuration(currentTime)}</span>
              <span className="time-separator">/</span>
              <span className="total-time">{formatDuration(audioInfo.duration)}</span>
            </div>

            <div className="progress-container" onClick={handleSeek}>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${(currentTime / audioInfo.duration) * 100}%` }} />
                <div className="progress-handle" style={{ left: `${(currentTime / audioInfo.duration) * 100}%` }} />
              </div>
            </div>

            {vizType === 'waveform' && (
              <div className="window-control">
                <label className="window-label">Window:</label>
                <select 
                  className="window-select"
                  value={windowSize}
                  onChange={(e) => setWindowSize(e.target.value === 'full' ? 'full' : Number(e.target.value))}
                >
                  <option value={1}>1 sec</option>
                  <option value={2}>2 sec</option>
                  <option value={3}>3 sec</option>
                  <option value={5}>5 sec</option>
                  <option value={10}>10 sec</option>
                  <option value="full">Full</option>
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

            {vizType === 'waveform' && (
              <button 
                className={`band-toggle-btn ${showBands ? 'active' : ''}`}
                onClick={() => setShowBands(!showBands)}
                title="Toggle frequency band separation"
              >
                🎛️ Bands
              </button>
            )}
          </div>
        )}

        {/* Frequency Band Controls */}
        {audioInfo && vizType === 'waveform' && showBands && (
          <div className="band-controls">
            <div className="band-controls-label">Frequency Bands:</div>
            <div className="band-toggles">
              {FREQUENCY_BANDS.map(band => (
                <label 
                  key={band.id} 
                  className={`band-checkbox ${visibleBands[band.id] ? 'active' : ''}`}
                  style={{ '--band-color': band.color }}
                >
                  <input
                    type="checkbox"
                    checked={visibleBands[band.id]}
                    onChange={(e) => setVisibleBands(prev => ({
                      ...prev,
                      [band.id]: e.target.checked
                    }))}
                  />
                  <span className="band-color-dot" style={{ background: band.color }}></span>
                  <span className="band-label">{band.shortLabel}</span>
                </label>
              ))}
            </div>
          </div>
        )}

        <div className="waveform-container">
          <div className="waveform-header">
            <h2 className="waveform-title">{vizLabel}</h2>
            {audioInfo && (
              <div className="waveform-info">
                {vizType === 'waveform' && (
                  <div className="info-badge">
                    Window: <span className="info-badge-value">{windowSize === 'full' ? 'Full' : `${windowSize}s`}</span>
                  </div>
                )}
                <div className="info-badge">
                  Sample Rate: <span className="info-badge-value">{audioInfo.sampleRate} Hz</span>
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

          {loading ? (
            <div className="loading-state">
              <div className="spinner"></div>
              <div className="loading-text">Processing audio on server...</div>
            </div>
          ) : audioInfo ? (
            <div className="canvas-container">
              <canvas ref={canvasRef} className="waveform-canvas"></canvas>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">🎧</div>
              <div className="empty-title">No Audio Loaded</div>
              <div className="empty-text">
                Upload an audio file to visualize. All processing happens on the server.
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
