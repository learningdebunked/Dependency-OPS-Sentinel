"""
Distributed training utilities (Experimental)
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, Callable, Dict, Any, List
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler

from .base_model import BaseModel
from .trainer import ModelTrainer
from .data.dataset import DOSDataset

def setup(rank: int, world_size: int):
    """
    Initialize the distributed environment.
    
    Args:
        rank: Process rank
        world_size: Number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """Clean up distributed processes."""
    dist.destroy_process_group()

class DistributedModelTrainer(ModelTrainer):
    """
    Distributed trainer using PyTorch's DistributedDataParallel.
    """
    
    def __init__(self, model: BaseModel, config: Dict[str, Any], rank: int, world_size: int):
        """
        Initialize the distributed trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            rank: Process rank
            world_size: Number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        # Move model to device and wrap with DDP
        model = model.to(self.device)
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[rank])
        
        super().__init__(model, config)
        
        # Adjust batch size for distributed training
        self.config['batch_size'] = self.config.get('batch_size', 32) // world_size
    
    def create_dataloader(self, dataset: DOSDataset, shuffle: bool = True) -> DataLoader:
        """
        Create a distributed data loader.
        
        Args:
            dataset: Dataset to load
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader with distributed sampling
        """
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch with distributed support."""
        self.model.train()
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.epoch)
            
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs['reconstructed'], batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Reduce loss across all processes
            reduced_loss = self._reduce_tensor(loss.detach())
            total_loss += reduced_loss.item()
        
        # Average loss across all processes
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model with distributed support."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.criterion(outputs['reconstructed'], batch)
                
                # Reduce loss across all processes
                reduced_loss = self._reduce_tensor(loss)
                total_loss += reduced_loss.item()
        
        # Average loss across all processes
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def _reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce tensor across all processes."""
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

def run_distributed_training(
    rank: int,
    world_size: int,
    model_fn: Callable[[], BaseModel],
    train_dataset: DOSDataset,
    val_dataset: DOSDataset,
    config: Dict[str, Any]
) -> None:
    """
    Run distributed training.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        model_fn: Function that returns a model instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
    """
    # Initialize distributed training
    setup(rank, world_size)
    
    try:
        # Create model and trainer
        model = model_fn()
        trainer = DistributedModelTrainer(model, config, rank, world_size)
        
        # Create data loaders
        train_loader = trainer.create_dataloader(train_dataset, shuffle=True)
        val_loader = trainer.create_dataloader(val_dataset, shuffle=False)
        
        # Train the model
        trainer.train(train_loader, val_loader)
        
    finally:
        # Clean up
        cleanup()

def launch_distributed_training(
    model_fn: Callable[[], BaseModel],
    train_dataset: DOSDataset,
    val_dataset: DOSDataset,
    config: Dict[str, Any],
    num_gpus: Optional[int] = None
) -> None:
    """
    Launch distributed training on multiple GPUs.
    
    Args:
        model_fn: Function that returns a model instance
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        num_gpus: Number of GPUs to use (default: all available)
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        raise ValueError("Distributed training requires at least 2 GPUs")
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Launch processes
    mp.spawn(
        run_distributed_training,
        args=(
            num_gpus,
            model_fn,
            train_dataset,
            val_dataset,
            config
        ),
        nprocs=num_gpus,
        join=True
    )
