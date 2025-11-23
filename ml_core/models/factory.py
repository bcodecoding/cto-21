"""Model factory for creating and managing different model architectures."""

import torch
import torch.nn as nn
from typing import Dict, Callable, Any


class SimpleCNN(nn.Module):
    """Simple CNN for tabular data reshaped as 1D sequences."""
    
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x


class SimpleMLPClassifier(nn.Module):
    """Simple Multi-Layer Perceptron for classification."""
    
    def __init__(self, input_size: int, num_classes: int, 
                 hidden_layers: list = None, dropout: float = 0.2):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64]
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ModelFactoryRegistry:
    """Registry for managing model creation functions."""
    
    def __init__(self):
        self._registry: Dict[str, Callable] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default model architectures."""
        self.register("simple_cnn", self._create_simple_cnn)
        self.register("simple_mlp", self._create_simple_mlp)
    
    def register(self, name: str, factory_fn: Callable):
        """
        Register a model factory function.
        
        Args:
            name: Model identifier
            factory_fn: Function that creates the model
        """
        self._registry[name] = factory_fn
    
    def create(self, name: str, **kwargs) -> nn.Module:
        """
        Create a model by name.
        
        Args:
            name: Model identifier
            **kwargs: Arguments to pass to the factory function
        
        Returns:
            PyTorch model
        """
        if name not in self._registry:
            raise ValueError(f"Model '{name}' not found in registry. "
                           f"Available models: {list(self._registry.keys())}")
        
        return self._registry[name](**kwargs)
    
    def list_models(self) -> list:
        """List all registered model names."""
        return list(self._registry.keys())
    
    @staticmethod
    def _create_simple_cnn(input_size: int, num_classes: int, 
                          hidden_size: int = 64, **kwargs) -> nn.Module:
        """Create a simple CNN model."""
        return SimpleCNN(input_size, num_classes, hidden_size)
    
    @staticmethod
    def _create_simple_mlp(input_size: int, num_classes: int,
                          hidden_layers: list = None, dropout: float = 0.2, 
                          **kwargs) -> nn.Module:
        """Create a simple MLP model."""
        return SimpleMLPClassifier(input_size, num_classes, hidden_layers, dropout)


def create_optimizer(model: nn.Module, optimizer_type: str = "adam",
                    learning_rate: float = 0.001, **kwargs) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer (adam, sgd, adamw)
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters
    
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == "sgd":
        momentum = kwargs.pop("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=learning_rate, 
                              momentum=momentum, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str = "step",
                    **kwargs) -> Any:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler (step, cosine, plateau)
        **kwargs: Additional scheduler parameters
    
    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == "step":
        step_size = kwargs.get("step_size", 10)
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type.lower() == "cosine":
        T_max = kwargs.get("T_max", 50)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type.lower() == "plateau":
        mode = kwargs.get("mode", "min")
        patience = kwargs.get("patience", 5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=patience
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
