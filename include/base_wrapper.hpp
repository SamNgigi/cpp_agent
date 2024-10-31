#ifndef _BASE_LLAMA_WRAPPER_HPP
#define _BASE_LLAMA_WRAPPER_HPP

#include <llama.h>

class BaseLlamaWrapper {
private:
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  llama_sampler* sampler = nullptr;

public:
  BaseLlamaWrapper(const char* model_path);
  ~BaseLlamaWrapper();
  double base_inference();
};


#endif //_BASE_LLAMA_WRAPPER_HPP
