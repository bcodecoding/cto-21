import React, { useState, useEffect } from 'react'
import ModelSelector from './components/ModelSelector'
import DatasetUploader from './components/DatasetUploader'
import TrainingMonitor from './components/TrainingMonitor'
import { api } from './api'
import './App.css'

function App() {
  const [models, setModels] = useState([])
  const [datasets, setDatasets] = useState([])
  const [runs, setRuns] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedDataset, setSelectedDataset] = useState('')
  const [selectedRunId, setSelectedRunId] = useState(null)
  const [training, setTraining] = useState(false)

  useEffect(() => {
    loadModels()
    loadDatasets()
    loadRuns()
    const interval = setInterval(loadRuns, 3000)
    return () => clearInterval(interval)
  }, [])

  const loadModels = async () => {
    try {
      const data = await api.getModels()
      setModels(data)
      if (data.length > 0) {
        setSelectedModel(data[0].id)
      }
    } catch (error) {
      console.error('Failed to load models:', error)
    }
  }

  const loadDatasets = async () => {
    try {
      const data = await api.getDatasets()
      setDatasets(data)
      if (data.length > 0) {
        setSelectedDataset(data[0].id)
      }
    } catch (error) {
      console.error('Failed to load datasets:', error)
    }
  }

  const loadRuns = async () => {
    try {
      const data = await api.getRuns()
      setRuns(data)
    } catch (error) {
      console.error('Failed to load runs:', error)
    }
  }

  const handleUploadDataset = async (file) => {
    try {
      const dataset = await api.uploadDataset(file)
      await loadDatasets()
      setSelectedDataset(dataset.id)
    } catch (error) {
      console.error('Failed to upload dataset:', error)
      alert('Failed to upload dataset')
    }
  }

  const handleStartTraining = async () => {
    if (!selectedModel || !selectedDataset) {
      alert('Please select both a model and dataset')
      return
    }

    setTraining(true)
    try {
      const run = await api.startTraining(selectedModel, selectedDataset)
      await loadRuns()
      setSelectedRunId(run.id)
    } catch (error) {
      console.error('Failed to start training:', error)
      alert('Failed to start training')
    } finally {
      setTraining(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>ML Training Platform</h1>
      </header>
      <main>
        <div className="config-section">
          <ModelSelector
            models={models}
            selectedModel={selectedModel}
            onSelect={setSelectedModel}
          />
          <DatasetUploader
            datasets={datasets}
            selectedDataset={selectedDataset}
            onSelect={setSelectedDataset}
            onUpload={handleUploadDataset}
          />
          <div className="card">
            <button
              className="train-button"
              onClick={handleStartTraining}
              disabled={training || !selectedModel || !selectedDataset}
            >
              {training ? 'Starting...' : 'Start Training'}
            </button>
          </div>
        </div>
        <div className="monitor-section">
          <TrainingMonitor
            runs={runs}
            selectedRunId={selectedRunId}
            onSelectRun={setSelectedRunId}
          />
        </div>
      </main>
    </div>
  )
}

export default App
