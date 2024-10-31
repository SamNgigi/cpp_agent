#include "base_wrapper.hpp"
#include "llama.h"

#include <chrono>
#include <print>

BaseLlamaWrapper::BaseLlamaWrapper(const char *model_path) {
  // Initialize model with optimal parameters
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = -1; // Auto-detect GPU layers

  model = llama_load_model_from_file(model_path, model_params);
  if (!model) {
    throw std::runtime_error("Failed to load model");
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;
  ctx_params.n_threads = 4;

  ctx = llama_new_context_with_model(model, ctx_params);
  if (!ctx) {
    llama_free_model(model);
    throw std::runtime_error("Failed to create context");
  }

  // Initialize sampler with temperature
  sampler = llama_sampler_init_greedy();
}

BaseLlamaWrapper::~BaseLlamaWrapper() {
  if (sampler) {
    llama_sampler_free(sampler);
  }
  llama_free(ctx);
  llama_free_model(model);
}

double BaseLlamaWrapper::base_inference() {

   auto start = std::chrono::high_resolution_clock::now();
  
  const char* prompt = "Tell me about machine learning";
  std::vector<llama_token> tokens(32);
  int n_tokens = llama_tokenize(model, prompt, -1, tokens.data(), tokens.size(), true, false);
  tokens.resize(n_tokens);

  // Process initial prompt
  llama_batch batch = llama_batch_get_one(
      tokens.data(),
      tokens.size()
  );

  if (llama_decode(ctx, batch) != 0) {
      throw std::runtime_error("Failed to decode");
  }

  // Generate tokens
  const int n_max_tokens = 512;
  std::vector<llama_token> output_tokens;
  output_tokens.reserve(n_max_tokens);

  for (int i = 0; i < n_max_tokens; ++i) {
      // Sample next token using the sampler
      llama_token new_token = llama_sampler_sample(sampler, ctx, i);
      
      if (new_token == llama_token_eos(model)) {
          break;
      }

      output_tokens.push_back(new_token);

      // Process the new token
      batch = llama_batch_get_one(&new_token, 1);
      if (llama_decode(ctx, batch) != 0) {
          throw std::runtime_error("Failed to decode");
      }
  }
    
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::println("WE GOT HERE");
  return diff.count();
}
