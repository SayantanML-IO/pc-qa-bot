"""
Device management utility for handling CUDA/CPU fallback gracefully.
"""
import logging
import torch
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "cpu"]


class DeviceManager:
    """Manages device selection with automatic CUDA/CPU fallback."""
    
    _instance = None
    _device: DeviceType = "cpu"
    _cuda_available: bool = False
    
    def __new__(cls):
        """Singleton pattern to ensure one device manager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize and detect available devices."""
        self._cuda_available = torch.cuda.is_available()
        
        if self._cuda_available:
            self._device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✓ CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            self._device = "cpu"
            logger.warning("⚠ CUDA not available. Falling back to CPU (slower performance)")
    
    @property
    def device(self) -> DeviceType:
        """Get current device ('cuda' or 'cpu')."""
        return self._device
    
    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available
    
    def get_device_kwargs(self) -> dict[str, str]:
        """Get device kwargs for model initialization."""
        return {"device": self._device}
    
    def log_device_info(self) -> None:
        """Log detailed device information."""
        if self._cuda_available:
            logger.info(f"Device: {self._device.upper()}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")
        else:
            logger.info(f"Device: {self._device.upper()}")
            logger.info(f"PyTorch Version: {torch.__version__}")


def get_device() -> DeviceType:
    """
    Get the current device (cuda or cpu).
    
    Returns:
        Device string: 'cuda' or 'cpu'
    """
    return DeviceManager().device


def get_device_kwargs() -> dict[str, str]:
    """
    Get device kwargs for model initialization.
    
    Returns:
        Dictionary with 'device' key
    """
    return DeviceManager().get_device_kwargs()