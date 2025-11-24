import asyncio
import random

from backend.models import RunStatus
from backend.storage import RunStore


class TrainerService:
    def __init__(self, run_store: RunStore):
        self.run_store = run_store

    async def train_model(self, run_id: str):
        """
        Simulates training a model. In a real scenario, this would
        interface with actual ML training libraries.
        """
        run = self.run_store.get_run(run_id)
        if not run:
            return

        try:
            # Update status to running
            self.run_store.update_status(run_id, RunStatus.RUNNING)
            self.run_store.append_log(
                run_id, f"Starting training for model: {run.model_id}"
            )
            self.run_store.append_log(run_id, f"Using dataset: {run.dataset_id}")
            self.run_store.append_log(run_id, f"Hyperparameters: {run.hyperparameters}")

            epochs = int(run.hyperparameters.get("epochs", 3))

            # Simulate training epochs
            for epoch in range(1, epochs + 1):
                await asyncio.sleep(2)  # Simulate training time

                # Simulate metrics
                loss = 1.0 - (epoch * 0.2) + random.uniform(-0.1, 0.1)
                accuracy = 0.5 + (epoch * 0.15) + random.uniform(-0.05, 0.05)
                loss = max(0.1, min(1.0, loss))
                accuracy = max(0.5, min(1.0, accuracy))

                metrics = {
                    f"epoch_{epoch}_loss": round(loss, 4),
                    f"epoch_{epoch}_accuracy": round(accuracy, 4),
                }

                self.run_store.update_metrics(run_id, metrics)
                self.run_store.append_log(
                    run_id,
                    f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}",
                )

            # Complete training
            self.run_store.append_log(run_id, "Training completed successfully!")
            self.run_store.update_status(run_id, RunStatus.COMPLETED)

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.run_store.append_log(run_id, error_msg)
            self.run_store.update_status(run_id, RunStatus.FAILED, error=error_msg)

    def start_training_async(self, run_id: str):
        """Launch training in the background"""
        asyncio.create_task(self.train_model(run_id))
