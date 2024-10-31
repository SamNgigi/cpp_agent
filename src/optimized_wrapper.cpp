#include "optimized_wrapper.hpp"
#include "llama.h"

#include <chrono>

OptimizedLlamaWrapper::OptimizedLlamaWrapper(const char* model_path){
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = -1;

  model = llama_load_model_from_file(model_path, model_params);
  if(!model){
    throw std::runtime_error("Failed to load model");
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;
  ctx_params.n_threads = 4;

  ctx = llama_new_context_with_model(model, ctx_params);
  if(!ctx){
    llama_free_model(model);
    throw std::runtime_error("Failed to create context");
  }

  sampler = llama_sampler_init_greedy();
}

OptimizedLlamaWrapper::~OptimizedLlamaWrapper(){
  if(sampler){llama_sampler_free(sampler);}
  llama_free(ctx);
  llama_free_model(model);
}

double OptimizedLlamaWrapper::optimized_inference(){
  auto start = std::chrono::high_resolution_clock::now();


  // Pre_allocate batch
  llama_batch batch = llama_batch_init(
    BATCH_SIZE, // n_tokens
    0,          // embd
    4           // n_seq_max
  );

  // Tokenize input
  std::vector<llama_token> tokens;
  tokens.resize(BATCH_SIZE);

  const char* prompt = "Tell me about machine learning";
  int n_tokens = llama_tokenize(
    model,
    prompt,
    -1,
    tokens.data(),
    BATCH_SIZE,
    true,
    false
  );
  tokens.resize(n_tokens);

  // Fill bach with tokens
  for(int i = 0; i < n_tokens; i++){
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = 1;
  }

  batch.n_tokens = n_tokens;

  // Process entire batch at once
  if(llama_decode(ctx, batch) != 0){
    throw std::runtime_error("Failed to decode batch");
  }


  // Generate tokens in batches
  const int n_max_tokens = 512;
  std::vector<llama_token> output_tokens;
  output_tokens.reserve(n_max_tokens);

  while(output_tokens.size() < n_max_tokens){
    // Fill batch with new tokens
    batch.n_tokens = 0;
    while(batch.n_tokens < BATCH_SIZE && output_tokens.size() < n_max_tokens){
      llama_token new_token = llama_sampler_sample(sampler, ctx, output_tokens.size());

      if(new_token == llama_token_eos(model)){break;}

      int pos = batch.n_tokens;
      batch.token[pos] = new_token;
      batch.pos[pos] = output_tokens.size() + n_tokens;
      batch.n_seq_id[pos] = 1;
      batch.seq_id[pos][0] = 0;
      batch.logits[pos] = 1;

      batch.n_tokens++;
      output_tokens.push_back(new_token);
    }
    if(batch.n_tokens == 0){break;}

    // Process batch
    if(llama_decode(ctx, batch) != 0){
      throw std::runtime_error("Failed to decode batch");
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  return diff.count();
}
