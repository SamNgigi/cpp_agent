#include "benchmark_wrapper.hpp"
#include "utils.hpp"

#include <print>
#include <string>
#include <llama.h>


int main(int argc, char** argv){

  std::print(stderr, "Hello World!!\n\n");


  if(argc != 2 || strstr("base,optimized", argv[1])==NULL){
    std::print(stderr, "Invalid run_type. Use 'base' or 'optimized'\n");
    std::print(stderr, "Usage: ./run.sh <run_type>\nrun_type: 'base' or 'optimized'\n");
    return EXIT_FAILURE;
  }

  std::string run_type = argv[1];


  auto llama_info = llama_print_system_info();
  std::print(stderr, "llama sytem info::\n{}\n\n", llama_info);
  std::unordered_map<std::string, std::string> env = llm_agent::utils::read_env(); 
  std::print(stderr, "model_path :: {}\n\n", env["CODE_LLAMA"]);


  try {
    BenchmarkWrapper benchmark(env["CODE_LLAMA"].c_str());
    benchmark.runner(run_type.c_str());
  }catch(const std::exception& e) {
    std::print(stderr, "Main::Error: {}\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
