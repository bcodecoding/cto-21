import React from 'react'

export default function ModelSelector({ models, selectedModel, onSelect }) {
  return (
    <div className="card">
      <h3>Model Presets</h3>
      <div className="options">
        {models.map((model) => (
          <label key={model.id} className="option">
            <input
              type="radio"
              name="model"
              value={model.id}
              checked={selectedModel === model.id}
              onChange={() => onSelect(model.id)}
            />
            <div>
              <strong>{model.name}</strong>
              <p>{model.description}</p>
              <small>Default hyperparameters: {JSON.stringify(model.hyperparameters)}</small>
            </div>
          </label>
        ))}
      </div>
    </div>
  )
}
