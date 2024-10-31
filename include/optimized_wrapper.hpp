#include <llama.h>

class OptimizedLlamaWrapper {
private:
  llama_model* model;
  llama_context* ctx;
  llama_sampler* sampler;
  static const int BATCH_SIZE = 512; 

public:
  OptimizedLlamaWrapper(const char* model_path);
  ~OptimizedLlamaWrapper();
  double optimized_inference();
};
