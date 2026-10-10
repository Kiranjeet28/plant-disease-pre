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
  const [analysis, setAnalysis] = useState(null)
  const [error, setError] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
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
    setAnalysis(null)
    setError('')
  }

  function clearFile() {
    requestRef.current += 1
    abortRef.current?.abort()
    setSelectedFile(null)
    setPreviewUrl('')
    setAnalysis(null)
    setError('')
    setIsAnalyzing(false)
    if (inputRef.current) inputRef.current.value = ''
  }

  async function analyzeImage() {
    if (!selectedFile || isAnalyzing) return
    const requestId = ++requestRef.current
    const controller = new AbortController()
    abortRef.current = controller
    setIsAnalyzing(true)
    setError('')
    setAnalysis(null)
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      if (!response.ok) {
        if (response.status === 413) throw new Error('That image is larger than the 10 MB limit.')
        if (response.status >= 500) throw new Error('The analysis service is unavailable.')
        throw new Error('The image could not be analyzed. Please try again.')
      }
      const data = await response.json()
      if (!data.final_decision || !Array.isArray(data.models)) {
        throw new Error('The analysis service returned an incomplete result.')
      }
      if (requestId === requestRef.current) setAnalysis(data)
    } catch (requestError) {
      if (requestError.name === 'AbortError' || requestId !== requestRef.current) return
      setError(requestError instanceof TypeError
        ? 'Could not reach the analysis service. Is the backend running?'
        : requestError.message)
    } finally {
      if (requestId === requestRef.current) setIsAnalyzing(false)
    }
  }

  return (
    <main className="app-shell">
      <section className="panel" aria-labelledby="page-title">
        <header className="dashboard-header">
          <div>
            <p className="eyebrow">PLANT HEALTH / MODEL ANALYSIS</p>
            <h1 id="page-title">Plant Disease Intelligence</h1>
            <p className="subtitle">Inspect the verified model evidence behind a plant-health screening result.</p>
          </div>
          <span className="verified-pill">Verified pipeline</span>
        </header>

        <div className="workspace">
          <UploadPanel
            inputRef={inputRef}
            selectedFile={selectedFile}
            previewUrl={previewUrl}
            isDragging={isDragging}
            setIsDragging={setIsDragging}
            chooseFile={chooseFile}
            clearFile={clearFile}
          />
          <SummaryPanel analysis={analysis} isAnalyzing={isAnalyzing} />
        </div>

        {error && <p className="error-message" role="alert">{error}</p>}
        <button className="predict-button" type="button" onClick={analyzeImage} disabled={!selectedFile || isAnalyzing}>
          {isAnalyzing ? 'Analyzing models…' : 'Analyze image'}
        </button>
        {analysis && <AnalysisDashboard analysis={analysis} />}
        <p className="disclaimer">Model output is a screening aid, not a substitute for local agricultural expertise.</p>
      </section>
    </main>
  )
}

function UploadPanel({ inputRef, selectedFile, previewUrl, isDragging, setIsDragging, chooseFile, clearFile }) {
  return (
    <div
      className={`upload-column ${isDragging ? 'is-dragging' : ''}`}
      onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => { event.preventDefault(); setIsDragging(false); chooseFile(event.dataTransfer.files?.[0]) }}
    >
      <input ref={inputRef} className="file-input" type="file" accept="image/jpeg,image/png" onChange={(event) => chooseFile(event.target.files?.[0])} aria-label="Choose a plant image" />
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
  )
}

function SummaryPanel({ analysis, isAnalyzing }) {
  const result = analysis?.final_decision
  const uncertain = result && (result.status !== 'ok' || !result.prediction_available)
  return (
    <div className="result-column" aria-live="polite" aria-atomic="true">
      <div className="result-heading"><p className="eyebrow">FINAL DECISION</p><h2>Router result</h2></div>
      {isAnalyzing ? <div className="empty-result loading-state"><span className="spinner" aria-hidden="true" />Running verified analysis…</div>
        : !result ? <p className="empty-result">Upload an image to inspect model-level evidence.</p>
          : <div className={`result-content result-${uncertain ? 'uncertain' : 'ok'}`}>
            <span className="status-badge">{uncertain ? 'Needs review' : result.is_healthy ? 'Healthy result' : 'Prediction available'}</span>
            <p className="result-label">{uncertain ? 'No reliable classification' : result.is_healthy ? 'Plant status' : 'Likely condition'}</p>
            <p className="disease-name">{result.display_label || result.user_message}</p>
            {result.confidence !== null && result.prediction_available && <ScoreBar label={result.calibration_status === 'calibrated' ? 'Calibrated confidence' : 'Uncalibrated model score'} score={result.confidence} />}
            <p className="notice">{result.explanation || result.user_message}</p>
          </div>}
    </div>
  )
}

function AnalysisDashboard({ analysis }) {
  const [showDecision, setShowDecision] = useState(true)
  const predictions = analysis.models.filter((model) => model.prediction)
  return (
    <div className="analysis-dashboard">
      <section className="dashboard-section">
        <div className="section-heading"><div><p className="eyebrow">MODEL EVIDENCE</p><h2>Individual model analysis</h2></div><span className="muted">{predictions.length} evaluated · {analysis.models.length - predictions.length} not evaluated</span></div>
        <div className="model-grid">{analysis.models.map((model) => <ModelCard key={model.model_id} model={model} />)}</div>
      </section>
     
      <section className="dashboard-section">
        <div className="section-heading"><div><p className="eyebrow">DECISION TRACE</p><h2>How the router decided</h2></div><button className="small-button" type="button" onClick={() => setShowDecision(!showDecision)}>{showDecision ? 'Collapse' : 'Expand'}</button></div>
        {showDecision && <DecisionTrace analysis={analysis} />}
      </section>
     
    </div>
  )
}

function ModelCard({ model }) {
  const [expanded, setExpanded] = useState(false)
  const prediction = model.prediction
  return (
    <article className={`model-card ${prediction ? 'evaluated' : 'excluded'}`}>
      <div className="model-card-header"><div><span className="model-status">{prediction ? 'Evaluated' : model.status.replace('_', ' ')}</span><h3>{model.model_id}</h3><p className="muted">{model.adapter_type} · {model.supported_crops.length ? `${model.supported_crops.length} supported crops` : 'No verified crops'}</p></div><button className="small-button" type="button" onClick={() => setExpanded(!expanded)}>{expanded ? 'Hide details' : 'Details'}</button></div>
      {prediction ? <><p className="model-prediction">{prediction.display_label || prediction.source_label || 'Unresolved result'}</p><p className="muted">Score type: {prediction.score_type || 'unavailable'} · {prediction.inference_duration_ms === null ? 'latency unavailable' : `${Math.round(prediction.inference_duration_ms)} ms`}</p><div className="top-list">{(prediction.top_predictions || []).slice(0, 3).map((item) => <div key={item.rank}><span>#{item.rank} {item.source_label}</span><b>{Number(item.score).toFixed(4)}</b></div>)}</div></>
        : <p className="notice">{model.exclusion_reason || 'Not evaluated.'}</p>}
      {expanded && <div className="details-panel"><p><strong>Observable process:</strong> model preprocessing → raw output → source-label mapping → canonical registry mapping → safety checks.</p><p><strong>Limitations:</strong> {model.limitations.join(' ') || 'None recorded.'}</p><p><strong>Supported crops:</strong> {model.supported_crops.join(', ') || 'None verified.'}</p>{prediction?.diagnostics?.artifact?.artifact_sha256 && <p><strong>Artifact SHA-256:</strong> <code>{prediction.diagnostics.artifact.artifact_sha256}</code></p>}</div>}
    </article>
  )
}

function DecisionTrace({ analysis }) {
  const result = analysis.final_decision
  return <ol className="decision-trace">
    {['Image received and validated', `${analysis.models.filter((model) => model.status === 'ready').length} ready model(s) identified`, `${analysis.models.filter((model) => model.prediction).length} model(s) evaluated`, 'Canonical mapping and safety checks applied', `Router result: ${result.status}`].map((step) => <li key={step}>{step}</li>)}
    <li className="decision-final">{result.explanation || result.user_message}</li>
  </ol>
}

function ScoreBar({ label, score }) {
  return <div className="confidence-block"><div className="confidence-row"><span>{label}</span><strong>{Math.round(score * 100)}%</strong></div><div className="confidence-track"><span style={{ width: `${Math.min(1, Math.max(0, score)) * 100}%` }} /></div></div>
}

function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>
}

export default App
