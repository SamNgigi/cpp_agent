#ifndef _BENCHMARK_WRAPPER_HPP
#define _BENCHMARK_WRAPPER_HPP

#include <llama.h>
#include <utility>

class BenchmarkWrapper {
private:
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  llama_sampler* sampler = nullptr;

public:
  BenchmarkWrapper(const char* model_path);
  ~BenchmarkWrapper();
  std::pair<double, int> run_base_inference();
  std::pair<double, int> run_optimized_inference();
  void runner(const char* run_type);
};



#endif //_BENCHMARK_WRAPPER_HPP
