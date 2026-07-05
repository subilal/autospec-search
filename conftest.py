import sys
import os
import warnings

# Add project root to sys.path so 'api' is importable
sys.path.insert(0, os.path.dirname(__file__))

# Suppress before any imports trigger them
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastapi")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")