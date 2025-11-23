import React, { useState } from 'react'

export default function DatasetUploader({ datasets, selectedDataset, onSelect, onUpload }) {
  const [uploading, setUploading] = useState(false)

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    setUploading(true)
    try {
      await onUpload(file)
      e.target.value = ''
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="card">
      <h3>Datasets</h3>
      <div className="upload-section">
        <input
          type="file"
          onChange={handleFileUpload}
          disabled={uploading}
          accept=".csv,.json,.txt"
        />
        {uploading && <span>Uploading...</span>}
      </div>
      <div className="options">
        {datasets.length === 0 ? (
          <p>No datasets available. Upload one above.</p>
        ) : (
          datasets.map((dataset) => (
            <label key={dataset.id} className="option">
              <input
                type="radio"
                name="dataset"
                value={dataset.id}
                checked={selectedDataset === dataset.id}
                onChange={() => onSelect(dataset.id)}
              />
              <div>
                <strong>{dataset.name}</strong>
                <small>Size: {(dataset.size / 1024).toFixed(2)} KB</small>
              </div>
            </label>
          ))
        )}
      </div>
    </div>
  )
}
