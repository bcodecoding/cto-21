const API_BASE = import.meta.env.PROD ? '' : 'http://localhost:8000';

export const api = {
  async getModels() {
    const response = await fetch(`${API_BASE}/models`);
    return response.json();
  },

  async getDatasets() {
    const response = await fetch(`${API_BASE}/datasets`);
    return response.json();
  },

  async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE}/datasets`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  },

  async startTraining(modelId, datasetId, hyperparameters = null) {
    const response = await fetch(`${API_BASE}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_id: modelId,
        dataset_id: datasetId,
        hyperparameters,
      }),
    });
    return response.json();
  },

  async getRuns() {
    const response = await fetch(`${API_BASE}/runs`);
    return response.json();
  },

  async getRun(runId) {
    const response = await fetch(`${API_BASE}/runs/${runId}`);
    return response.json();
  },

  async runInference(runId, inputData) {
    const response = await fetch(`${API_BASE}/inference`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        run_id: runId,
        input_data: inputData,
      }),
    });
    return response.json();
  },
};
