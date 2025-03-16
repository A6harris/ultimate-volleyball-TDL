"""
Script to register the TD Learning trainer with ML-Agents.
This version is more flexible with different ML-Agents versions.
"""

import sys
import os
import importlib
from typing import Dict, Type, Callable, Any

def check_mlagents_version():
    """Check the installed ML-Agents version and print information."""
    try:
        import mlagents
        print(f"ML-Agents version: {mlagents.__version__}")
        return True
    except ImportError:
        print("ML-Agents package not found. Please install it with:")
        print("pip install mlagents")
        return False
    except AttributeError:
        print("Found ML-Agents package but couldn't determine version.")
        print("Proceeding with registration attempt...")
        return True

def register_td_trainer_v1():
    """
    Register TD trainer with ML-Agents using the trainer_factory approach.
    This is for newer versions of ML-Agents.
    """
    try:
        from mlagents.trainers.trainer_factory import TrainerFactory
        from td_module import get_td_trainer
        from mlagents.trainers.settings import TrainerType
        
        # Add TD to the TrainerType enum if it doesn't exist
        if not hasattr(TrainerType, "TD"):
            TrainerType.TD = "td"
        
        # Register the TD trainer
        TrainerFactory.trainer_factory_dict[TrainerType.TD] = get_td_trainer
        print("Successfully registered TD trainer with TrainerFactory!")
        return True
    except ImportError:
        print("Couldn't register using trainer_factory approach.")
        return False

def register_td_trainer_v2():
    """
    Alternative registration method for older versions of ML-Agents.
    """
    try:
        import mlagents.trainers as trainers
        from td_module import get_td_trainer
        
        # Try to find the registry module or class
        if hasattr(trainers, "trainer_registry"):
            registry = trainers.trainer_registry
            if hasattr(registry, "register"):
                registry.register("td", get_td_trainer)
                print("Successfully registered TD trainer with trainer_registry!")
                return True
        
        # Try direct monkey patching if registry not found
        if hasattr(trainers, "registry"):
            trainers.registry["td"] = get_td_trainer
            print("Successfully registered TD trainer by patching registry!")
            return True
            
        print("Could not find trainer registry. Registration may not be successful.")
        return False
    except (ImportError, AttributeError) as e:
        print(f"Error in alternative registration: {e}")
        return False

def check_td_module():
    """Check if the TD module is available."""
    try:
        import td_module
        print("TD module found.")
        return True
    except ImportError:
        print("TD module not found. Make sure td_module.py is in the current directory.")
        return False

if __name__ == "__main__":
    print("Checking ML-Agents installation...")
    if not check_mlagents_version():
        sys.exit(1)
        
    print("\nChecking TD module...")
    if not check_td_module():
        sys.exit(1)
    
    print("\nAttempting to register TD trainer...")
    success = register_td_trainer_v1() or register_td_trainer_v2()
    
    if success:
        print("\nRegistration successful! You can now use trainer_type: td in your configuration.")
    else:
        print("\nRegistration failed. Your ML-Agents version might not support custom trainers.")
        print("Consider using the standalone TD implementation instead.")
    
    sys.exit(0 if success else 1)