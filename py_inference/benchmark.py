import time
import csv
import matplotlib.pyplot as plt
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
        n_runs: int = 1,
        output_dir: str = "benchmark_results"
    ):
        self.model_path = os.getenv('CODE_LLAMA')
        
        if not self.model_path:
            raise ValueError("MODEL PATH not found in env variables")

        self.n_runs = n_runs

        # Create output directory in py_inference folder
        self.output_dir = Path(__file__).parent / output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Initialize CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        self._init_csv()

        self.cpp_executable = Path(__file__).parent.parent / 'build' / 'Release' / 'cpp_agent.exe'
        if not self.cpp_executable.exists():
            raise FileNotFoundError(f"C++ executable not found at {self.cpp_executable}")

    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run_id', 'implementation', 'inference_time', 'tokens_generated'])

    def _save_result(self, run_id: int, implementation: str, time: float, tokens: int):
        """Save a single benchmark result to CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([run_id, implementation, time, tokens])


    def run_python_inference(self) -> List[float]:
        """Run python inference benchmark"""
        model = Llama(self.model_path)
        times = []


        for i in range(self.n_runs):
            print(f"\nRunning Python inference iteration {i+1}/{self.n_runs}")
            start = time.perf_counter()
            
            output = model(
                "Tell me about machine learning",
                max_tokens=512,
                temperature=1.0,  # Increase temperature
                top_p=1.0,        # Consider all tokens
                top_k=0,          # Disable top-k filterin
            )

            end = time.perf_counter()
            inference_time = end - start
            tokens_generated = len(output['choices'][0]['text'])
            
            self._save_result(i, 'python', inference_time, tokens_generated)
            times.append(inference_time)
            print(f"Successfully recorded: time={inference_time:.4f}s, tokens={tokens_generated}")

        return times
    
    def _get_enhanced_environment(self) -> Dict[str, str]:
        """Create an enhanced environment with necessary DLL paths"""
        env = os.environ.copy()
        
        # Add the directory containing the required DLLs to the PATH
        dll_directory = r'C:\opt\llama.cpp\bin'  # Use raw string to handle backslashes
        
        # Prepend the DLL directory to the PATH
        env['PATH'] = dll_directory + os.pathsep + env.get('PATH', '')
        
        # Set CODE_LLAMA environment variable
        env['CODE_LLAMA'] = self.model_path
        
        return env


    def run_cpp_base_inference(self) -> List[float]:
        """Run C++ base inference benchmark"""
        times = []
        enhanced_env = self._get_enhanced_environment()

        
        for i in range(self.n_runs):
            try:
                print(f"\nRunning C++ base inference iteration {i+1}/{self.n_runs}")
                
                result = subprocess.run(
                    [str(self.cpp_executable), 'base'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=enhanced_env,
                    cwd=str(self.cpp_executable.parent.parent.parent)  # Run from executable directory
                )
                
                if result.returncode != 0:
                    error_msg = (
                        f"C++ inference failed (code: {result.returncode})\n"
                        f"STDOUT: {result.stdout}\n"
                        f"STDERR: {result.stderr}"
                    )
                    raise RuntimeError(error_msg)
                
                # Parse output
                inference_time_str, tokens_str = result.stdout.strip().split(',')
                inference_time = float(inference_time_str)
                tokens = int(tokens_str)
                self._save_result(i, 'cpp_base', inference_time, tokens)
                times.append(inference_time)
                
                print(f"Successfully recorded: time={inference_time:.4f}s, tokens={tokens}")
                
            except Exception as e:
                print(f"Error in iteration {i+1}: {str(e)}")
                raise
        
        return times

    def run_cpp_optimized_inference(self) -> List[float]:
        """Run C++ base inference benchmark"""
        times = []
        enhanced_env = self._get_enhanced_environment()

        
        for i in range(self.n_runs):
            try:
                print(f"\nRunning C++ optimized inference iteration {i+1}/{self.n_runs}")
                
                result = subprocess.run(
                    [str(self.cpp_executable), 'optimized'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=enhanced_env,
                    cwd=str(self.cpp_executable.parent.parent.parent)  # Run from executable directory
                )
                
                if result.returncode != 0:
                    error_msg = (
                        f"C++ inference failed (code: {result.returncode})\n"
                        f"STDOUT: {result.stdout}\n"
                        f"STDERR: {result.stderr}"
                    )
                    raise RuntimeError(error_msg)
                
                # Parse output
                inference_time_str, tokens_str = result.stdout.strip().split(',')
                inference_time = float(inference_time_str)
                tokens = int(tokens_str)
                self._save_result(i, 'cpp_optimized', inference_time, tokens)
                times.append(inference_time)
                
                print(f"Successfully recorded: time={inference_time:.4f}s, tokens={tokens}")
                
            except Exception as e:
                print(f"Error in iteration {i+1}: {str(e)}")
                raise
        
        return times


    def run_all_benchmarks(self):
        """Run all benchmark implementations"""
        print(f"Using model path: {self.model_path}\n")
        print(f"Results with be saved to: {self.output_dir}\n")

        print("Running python implementation...\n")
        python_times = self.run_python_inference()

        print("Running C++ base implementation...\n")
        cpp_base_times = self.run_cpp_base_inference()
        
        print("Running C++ base implementation...\n")
        cpp_optimized_times = self.run_cpp_optimized_inference()
        
        return python_times, cpp_base_times, cpp_optimized_times


if __name__ == "__main__":
    
    try:
        benchmark = InferenceBenchmark(n_runs = 20)
        benchmark.run_all_benchmarks()
        # benchmark.run_python_inference()
    except Exception as e:
        print(f"Error: {e}")





