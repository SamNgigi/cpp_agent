import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from pathlib import Path
from llama_cpp import Llama
import subprocess
from datetime import datetime
import os
from dotenv import load_dotenv



# Loading environment variables from the parent directory
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

class InferenceBenchmark:
    def __init__(
        self,
        n_runs: int = 10,
        output_dir: str = "benchmark_results"
    ):
        pass
