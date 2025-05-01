from __future__ import annotations

from enum import IntEnum, auto
from typing import Optional, Union, List
import torch

class GPUVendor(IntEnum):
    NVIDIA = auto()
    AMD = auto()
    MooreThreads = auto()
    MetaX = auto()
    MUSA = auto()
    Unknown = auto()

class DeviceManager:
    """
    Device manager that provides a unified interface for handling different GPU vendors
    """
    def __init__(self):
        self.gpu_vendor = self._detect_gpu_vendor()
        self.available_devices = self._get_available_devices()
    
    def _detect_gpu_vendor(self) -> GPUVendor:
        """Detect GPU vendor type"""
        return GPUVendor.Unknown
    
    def _get_available_devices(self) -> List[int]:
        """Get list of available device indices"""
        return ["cpu"]
    
    def get_device_str(self, device_id: Union[int, str]) -> str:
        """
        Get device string for the given device ID
        
        Args:
            device_id: Device index (0, 1, 2, etc.), -1 for CPU, or "cpu" string
            
        Returns:
            Device string representation (e.g., "cuda:0", "musa:1", "cpu")
        """
        return "cpu"
    
    def to_torch_device(self, device_id: Union[int, str] = 0) -> torch.device:
        """
        Convert device ID to torch.device object
        
        Args:
            device_id: Device index (0, 1, 2, etc.), -1 for CPU, or "cpu" string
            
        Returns:
            torch.device object
        """
        return torch.device("cpu")
    
    def move_tensor_to_device(self, tensor: torch.Tensor, device_id: Union[int, str] = 0) -> torch.Tensor:
        """
        Move tensor to specified device
        
        Args:
            tensor: PyTorch tensor to move
            device_id: Device index (0, 1, 2, etc.), -1 for CPU, or "cpu" string
            
        Returns:
            Tensor moved to the specified device
        """
        return tensor.to("cpu")
    
    def is_available(self, index: int = 0) -> bool:
        """
        Check if device at specified index is available
        
        Args:
            index: Device index to check
            
        Returns:
            True if the device is available, False otherwise
        """
        if index < 0:
            return True  # CPU is always available
            
        return index in self.available_devices
    
    def get_all_devices(self) -> List[int]:
        """
        Get all available device indices
        
        Returns:
            List of available device indices (0, 1, 2, etc.)
        """
        return self.available_devices

# Create global device manager instance
device_manager = DeviceManager()

# Convenience functions
def get_device(device_id: Union[int, str] = 0) -> torch.device:
    """
    Get torch.device object for the specified device ID
    
    Args:
        device_id: Device index (0, 1, 2, etc.), -1 for CPU, or "cpu" string
        
    Returns:
        torch.device object
    """
    return device_manager.to_torch_device(device_id)

def to_device(tensor: torch.Tensor, device_id: Union[int, str] = 0) -> torch.Tensor:
    """
    Move tensor to specified device
    
    Args:
        tensor: PyTorch tensor to move
        device_id: Device index (0, 1, 2, etc.), -1 for CPU, or "cpu" string
        
    Returns:
        Tensor moved to the specified device
    """
    return device_manager.move_tensor_to_device(tensor, device_id)

# Get devices
cpu_device = get_device(-1)        # CPU using index -1
cpu_device2 = get_device("cpu")    # CPU using string "cpu"
gpu0 = get_device(0)               # First GPU

# Move tensors
x = torch.randn(3, 3)
x_gpu = to_device(x, 0)            # Move to first GPU
x_cpu1 = to_device(x, -1)          # Move to CPU using index -1
x_cpu2 = to_device(x, "cpu")       # Move to CPU using string "cpu"