import React, { useEffect } from 'react'

export default function TrainingMonitor({ runs, selectedRunId, onSelectRun }) {
  useEffect(() => {
    if (!selectedRunId && runs.length > 0) {
      onSelectRun(runs[0].id)
    }
  }, [runs, selectedRunId, onSelectRun])

  const selectedRun = runs.find((run) => run.id === selectedRunId)

  return (
    <div className="card">
      <h3>Training Runs</h3>
      <div className="run-list">
        {runs.length === 0 ? (
          <p>No runs yet. Start one above.</p>
        ) : (
          runs.map((run) => (
            <button
              key={run.id}
              className={`run-item ${selectedRunId === run.id ? 'active' : ''}`}
              onClick={() => onSelectRun(run.id)}
            >
              <div>
                <strong>{run.model_id}</strong>
                <small>{run.id.slice(0, 8)}</small>
              </div>
              <span className={`status ${run.status.toLowerCase()}`}>{run.status}</span>
            </button>
          ))
        )}
      </div>

      {selectedRun && (
        <div className="run-details">
          <h4>Run Details</h4>
          <p><strong>Model:</strong> {selectedRun.model_id}</p>
          <p><strong>Dataset:</strong> {selectedRun.dataset_id}</p>
          <p><strong>Status:</strong> {selectedRun.status}</p>
          <p><strong>Hyperparameters:</strong> {JSON.stringify(selectedRun.hyperparameters)}</p>
          <div className="logs">
            <h5>Logs</h5>
            <pre>{selectedRun.logs.join('\n')}</pre>
          </div>
          {selectedRun.metrics && Object.keys(selectedRun.metrics).length > 0 && (
            <div className="metrics">
              <h5>Metrics</h5>
              <ul>
                {Object.entries(selectedRun.metrics).map(([key, value]) => (
                  <li key={key}>
                    <strong>{key}:</strong> {value}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
