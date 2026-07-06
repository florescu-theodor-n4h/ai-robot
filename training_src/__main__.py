"""Module entry point - delegates to main runLLMAgentForAgenticDevs.

Run as: python -m training_src
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # Import and run the main module
    import runLLMAgentForAgenticDevs
