import { useEffect, useRef, useState } from 'react'
import './App.css'

export const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
export const MAX_FILE_BYTES = 10 * 1024 * 1024
const ACCEPTED_TYPES = new Set(['image/jpeg', 'image/png'])

export function validateImageFile(file) {
  if (!file) return 'Please choose an image.'
  if (!ACCEPTED_TYPES.has(file.type)) return 'Please choose a JPG, JPEG, or PNG image.'
  if (file.size > MAX_FILE_BYTES) return 'Images must be 10 MB or smaller.'
  return ''
}

function App() {
  const inputRef = useRef(null)
  const requestRef = useRef(0)
  const abortRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isPredicting, setIsPredicting] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl])

  function chooseFile(file) {
    const validationError = validateImageFile(file)
    if (validationError) {
      setError(validationError)
      return
    }
    requestRef.current += 1
    abortRef.current?.abort()
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError('')
  }

  function selectFile(event) {
    chooseFile(event.target.files?.[0])
  }

  function handleDrop(event) {
    event.preventDefault()
    setIsDragging(false)
    chooseFile(event.dataTransfer.files?.[0])
  }

  function clearFile() {
    requestRef.current += 1
    abortRef.current?.abort()
    setSelectedFile(null)
    setPreviewUrl('')
    setResult(null)
    setError('')
    setIsPredicting(false)
    if (inputRef.current) inputRef.current.value = ''
  }

  async function predictDisease() {
    if (!selectedFile || isPredicting) return
    const requestId = ++requestRef.current
    const controller = new AbortController()
    abortRef.current = controller
    setIsPredicting(true)
    setError('')
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      const response = await fetch(`${API_URL}/predict/router`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      if (!response.ok) {
        if (response.status === 413) throw new Error('That image is larger than the 10 MB limit.')
        if (response.status === 400 || response.status === 422) throw new Error('The image could not be processed.')
        if (response.status >= 500) throw new Error('The prediction service is unavailable.')
        throw new Error('Prediction failed. Please try again.')
      }
      const data = await response.json()
      if (requestId === requestRef.current) setResult(data)
    } catch (requestError) {
      if (requestError.name === 'AbortError') return
      if (requestId !== requestRef.current) return
      setError(requestError instanceof TypeError
        ? 'Could not reach the prediction service. Is the backend running?'
        : requestError.message)
    } finally {
      if (requestId === requestRef.current) setIsPredicting(false)
    }
  }

  const confidence = typeof result?.confidence === 'number'
    ? Math.max(0, Math.min(result.confidence, 1)) * 100
    : null
  const resultTitle = result?.is_healthy ? 'Healthy plant' : result?.display_label

  return (
    <main className="app-shell">
      <section className="panel" aria-labelledby="page-title">
        <div className="intro">
          <p className="eyebrow">PLANT HEALTH / GUIDED ANALYSIS</p>
          <h1 id="page-title">Plant Disease Detection</h1>
          <p className="subtitle">Upload a clear leaf image to receive a model-assisted screening result.</p>
        </div>
        <div className="workspace">
          <div
            className={`upload-column ${isDragging ? 'is-dragging' : ''}`}
            onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <input ref={inputRef} className="file-input" type="file" accept="image/jpeg,image/png" onChange={selectFile} aria-label="Choose a plant image" />
            <button className="upload-button" type="button" onClick={() => inputRef.current?.click()}>
              <span className="upload-icon" aria-hidden="true">↑</span>
              <span>{selectedFile ? 'Choose another image' : 'Choose plant image'}</span>
            </button>
            <p className="drop-hint">or drag and drop here</p>
            <p className="file-hint">JPG, JPEG, or PNG · max 10 MB</p>
            {previewUrl && (
              <div className="preview-wrap">
                <img className="preview" src={previewUrl} alt={`Preview of ${selectedFile?.name || 'selected plant leaf'}`} />
                <p className="file-name">{selectedFile?.name}</p>
                <button className="clear-button" type="button" onClick={clearFile}>Clear image</button>
              </div>
            )}
          </div>
          <div className="result-column" aria-live="polite" aria-atomic="true">
            <div className="result-heading">
              <p className="eyebrow">RESULT</p>
              <h2>Prediction</h2>
            </div>
            {result ? (
              <ResultPanel result={result} title={resultTitle} confidence={confidence} />
            ) : isPredicting ? (
              <div className="empty-result loading-state"><span className="spinner" aria-hidden="true" />Analyzing your image…</div>
            ) : <p className="empty-result">Your prediction will appear here. Upload a clear leaf image to begin.</p>}
          </div>
        </div>
        {error && <p className="error-message" role="alert">{error}</p>}
        <button className="predict-button" type="button" onClick={predictDisease} disabled={!selectedFile || isPredicting}>
          {isPredicting ? 'Analyzing…' : 'Predict disease'}
        </button>
        <p className="disclaimer">This is an image-based screening aid, not a substitute for local agricultural expertise.</p>
      </section>
    </main>
  )
}

function ResultPanel({ result, title, confidence }) {
  const isUncertain = result.status !== 'ok' || result.abstained || !result.prediction_available
  const warnings = [...(result.image_quality_warnings || []), ...(result.ood_warning ? [result.ood_warning] : []), ...(result.warnings || [])]
  return (
    <div className={`result-content result-${isUncertain ? 'uncertain' : 'ok'}`}>
      <span className="status-badge">{isUncertain ? 'Needs review' : result.is_healthy ? 'Healthy result' : 'Prediction available'}</span>
      <p className="result-label">{isUncertain ? 'No reliable classification' : result.is_healthy ? 'Plant status' : 'Likely condition'}</p>
      <p className="disease-name">{title || result.user_message || 'No reliable result available'}</p>
      {confidence !== null && !isUncertain && (
        <>
          <div className="confidence-row"><span>{result.calibration_status === 'calibrated' ? 'Calibrated confidence' : 'Uncalibrated model score'}</span><strong>{Math.round(confidence)}%</strong></div>
          <div className="confidence-track" aria-label={`${Math.round(confidence)} percent ${result.calibration_status === 'calibrated' ? 'calibrated confidence' : 'uncalibrated model score'}`}><span style={{ width: `${confidence}%` }} /></div>
        </>
      )}
      <dl className="result-details">
        {result.calibration_status && <><dt>Calibration</dt><dd>{result.calibration_status.replaceAll('_', ' ')}</dd></>}
        {result.selected_model_id && <><dt>Model</dt><dd>{result.selected_model_id}</dd></>}
        {result.inference_time_ms !== null && <><dt>Inference</dt><dd>{Math.round(result.inference_time_ms)} ms</dd></>}
      </dl>
      {isUncertain && <p className="notice">{result.user_message} {result.abstention_reason ? `Reason: ${result.abstention_reason.replaceAll('_', ' ')}.` : ''}</p>}
      {warnings.length > 0 && <div className="warning-box"><strong>Image or model notes</strong><ul>{warnings.map((warning) => <li key={warning}>{warning.replaceAll('_', ' ')}</li>)}</ul></div>}
    </div>
  )
}

export default App
