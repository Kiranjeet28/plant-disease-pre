import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL

function App() {
  const inputRef = useRef(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [isPredicting, setIsPredicting] = useState(false)

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl])

  function selectFile(event) {
    const file = event.target.files?.[0]
    if (!file) return
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setError('Please choose a JPG, JPEG, or PNG image.')
      return
    }
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError('')
  }

  function clearFile() {
    setSelectedFile(null)
    setPreviewUrl('')
    setResult(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  async function predictDisease() {
    if (!selectedFile || isPredicting) return
    setIsPredicting(true)
    setError('')
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData })
      if (!response.ok) {
        if (response.status === 400) throw new Error('The image could not be processed.')
        if (response.status === 422) throw new Error('Please select a valid image file.')
        if (response.status >= 500) throw new Error('The prediction service is unavailable.')
        throw new Error('Prediction failed. Please try again.')
      }
      const data = await response.json()
      const confidence = Number(data.confidence)
      setResult({
        disease: data.disease,
        confidence: confidence <= 1 ? confidence * 100 : confidence,
        classIndex: data.class_index,
      })
    } catch (requestError) {
      setError(requestError instanceof TypeError
        ? 'Could not reach the prediction service. Is the backend running?'
        : requestError.message)
    } finally {
      setIsPredicting(false)
    }
  }

  return (
    <main className="app-shell">
      <section className="panel">
        <div className="intro">
          <p className="eyebrow">PLANT HEALTH / IMAGE CLASSIFICATION</p>
          <h1>Plant Disease Detection</h1>
          <p className="subtitle">Upload a leaf image to identify a likely disease.</p>
        </div>
        <div className="workspace">
          <div className="upload-column">
            <input ref={inputRef} className="file-input" type="file" accept="image/jpeg,image/png" onChange={selectFile} aria-label="Choose a plant image" />
            <button className="upload-button" type="button" onClick={() => inputRef.current?.click()}>
              <span className="upload-icon" aria-hidden="true">+</span>
              <span>{selectedFile ? 'Choose another image' : 'Choose plant image'}</span>
            </button>
            <p className="file-hint">JPG, JPEG, or PNG</p>
            {previewUrl && (
              <div className="preview-wrap">
                <img className="preview" src={previewUrl} alt="Selected plant leaf" />
                <button className="clear-button" type="button" onClick={clearFile}>Clear image</button>
              </div>
            )}
          </div>
          <div className="result-column" aria-live="polite">
            <div className="result-heading">
              <p className="eyebrow">RESULT</p>
              <h2>Prediction</h2>
            </div>
            {result ? (
              <div className="result-content">
                <p className="result-label">Detected disease</p>
                <p className="disease-name">{result.disease}</p>
                <div className="confidence-row"><span>Confidence</span><strong>{Math.round(result.confidence)}%</strong></div>
                <div className="confidence-track"><span style={{ width: `${Math.min(result.confidence, 100)}%` }} /></div>
              </div>
            ) : <p className="empty-result">Your prediction will appear here.</p>}
          </div>
        </div>
        {error && <p className="error-message" role="alert">{error}</p>}
        <button className="predict-button" type="button" onClick={predictDisease} disabled={!selectedFile || isPredicting}>
          {isPredicting ? 'Predicting...' : 'Predict disease'}
        </button>
      </section>
    </main>
  )
}

export default App
