from abc import ABC, abstractmethod
import os
import torch
from safetensors import safe_open
from typing import Dict, Any, Optional, Union
from ktransformers.util.custom_gguf import GGUFLoader

class ModelLoader(ABC):
    """
    Abstract base class for model loaders.
    Defines the interface that all model loaders must implement.
    """
    
    @abstractmethod
    def load_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """
        Load a tensor by name.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The loaded tensor
        """
        pass
    
    @classmethod
    @abstractmethod
    def supports_format(cls, path: str) -> bool:
        """
        Check if this loader supports the given path format.
        
        Args:
            path: Path to check
            
        Returns:
            True if this loader supports the given path, False otherwise
        """
        pass


class SafeTensorLoader(ModelLoader):
    """
    Loader for SafeTensor format models.
    """
    
    def __init__(self, path: str):
        """
        Initialize the SafeTensor loader.
        
        Args:
            path: Path to the model directory or file
        """
        self.tensor_file_map = {}  # Maps tensor names to file paths
        self.file_handle_map = {}  # Maps file names to file handles
        self._load_tensor_file_map(path)
    
    def _load_tensor_file_map(self, path: str) -> None:
        """
        Load the tensor file map from the given path.
        
        Args:
            path: Path to the model directory or file
        """
        # Normalize path to directory
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        if os.path.isfile(path):
            folder_path = os.path.dirname(path)
        else:
            folder_path = path

        found_safetensor = False
        for root, _, files in os.walk(folder_path):
            files = sorted(files)
            for file in files:
                if file.endswith(".safetensors"):
                    found_safetensor = True
                    file_path = os.path.join(root, file)
                    if file not in self.file_handle_map:
                        try:
                            handle = safe_open(file_path, framework="pt")
                            self.file_handle_map[file] = handle
                        except Exception as e:
                            print(f"Error opening Safetensor file {file_path}: {e}")
                            continue

                    f = self.file_handle_map.get(file)
                    if f is None:
                        continue
                    try:
                        for key in f.keys():
                            self.tensor_file_map[key] = file
                    except Exception as e:
                        print(f"Error reading Safetensor file {file_path}: {e}")

        if not found_safetensor:
            # Not raising an error here allows for the factory to try other loaders
            print(f"No Safetensor files found in {folder_path}")
    
    def load_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """
        Load a tensor by name.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The loaded tensor
        """
        if name not in self.tensor_file_map:
            raise KeyError(f"Key {name} not found in Safetensor files")
        file = self.tensor_file_map[name]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(name)
        return tensor.to(device)
    
    def load_dequantized_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """
        Load and dequantize a tensor.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The dequantized tensor
        """
        if name not in self.tensor_file_map:
            raise KeyError(f"Key {name} not found in Safetensor files")
        file = self.tensor_file_map[name]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(name).to(device)
        if name.endswith(".weight"):
            if name[:-7] + ".weight_scale_inv" in self.tensor_file_map:
                weight_scale_inv = f.get_tensor(name[:-7] + ".weight_scale_inv").to(device)
                # Assuming weight_dequant function is imported
                from ktransformers.ktransformers_ext.triton.fp8gemm import weight_dequant
                tensor = weight_dequant(tensor, weight_scale_inv)
        return tensor.to(device)
    
    def close_all_handles(self) -> None:
        """
        Close all file handles.
        """
        for handle in self.file_handle_map.values():
            handle.close()
        self.file_handle_map.clear()

    @classmethod
    def supports_format(cls, path: str) -> bool:
        """
        Check if this loader supports the given path format.
        
        Args:
            path: Path to check
            
        Returns:
            True if safetensor files are found in the path, False otherwise
        """
        # Normalize path to directory
        if not os.path.exists(path):
            return False
        if os.path.isfile(path):
            if path.endswith(".safetensors"):
                return True
            folder_path = os.path.dirname(path)
        else:
            folder_path = path
            
        # Check if any safetensor files exist in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".safetensors"):
                    return True
        return False


class ModelLoaderFactory:
    """
    Factory class for creating appropriate model loaders based on file format.
    """
    
    @classmethod
    def _gguf_supports_format(cls, path: str) -> bool:
        """Check if path contains GGUF files"""
        if not os.path.exists(path):
            return False
        if os.path.isfile(path):
            return path.endswith(".gguf")
        # Check directory for GGUF files
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".gguf"):
                    return True
        return False
    
    @classmethod
    def create_loader(cls, path: str):
        """
        Create an appropriate loader based on the given path.
        
        Args:
            path: Path to the model directory or file
            
        Returns:
            An instance of the appropriate loader
            
        Raises:
            ValueError: If no suitable loader is found for the given path
        """
        # Check for GGUF format first
        if cls._gguf_supports_format(path):
            return GGUFLoader(path)
        # Check for SafeTensor format
        elif SafeTensorLoader.supports_format(path):
            return SafeTensorLoader(path)
        else:
            raise ValueError(f"No suitable loader found for path: {path}")