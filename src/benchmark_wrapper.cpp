#include "benchmark_wrapper.hpp"
#include <llama.h>

#include <chrono>
#include <print>


BenchmarkWrapper::BenchmarkWrapper(const char* model_path){
  // Initialize model
  std::print("Initializing model from path: {}\n\n", model_path);
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0; // CPU only for testing
  model_params.vocab_only = false;
  model = llama_load_model_from_file(model_path, model_params);
  if(!model){ throw std::runtime_error("Failed to load model"); }
  std::print("\nModel loaded successfully\n\n");

  
  // Initialize context
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;
  ctx_params.n_threads = 4;
  ctx_params.logits_all = false;
  ctx  = llama_new_context_with_model(model, ctx_params);
  if(!ctx){
    llama_free_model(model);
    throw std::runtime_error("Failed to create context");
  }
  std::print("\nContext loaded successfully\n");

  // Intialize sampler
  sampler = llama_sampler_init_greedy();
  if(!sampler){
    llama_free(ctx);
    llama_free_model(model);
    throw std::runtime_error("Failed to initialize sampler");
  }
  std::print("\nSampler initialized successfully\n");
}

std::vector<llama_token> tokenize_input(llama_model* model, const char* prompt){
  const int text_len = strlen(prompt);
  // Estimate maximum possible tokens (conservatively)
  const int max_tokens = text_len + 2; // + 2 for potential BOS and EOS tokens
  std::vector<llama_token> tokens(max_tokens);
  int n_tokens = llama_tokenize(
    model,
    prompt,
    text_len,
    tokens.data(),
    max_tokens,
    true,
    false
  );
  if(n_tokens < 0){
    throw std::runtime_error("Tokenization failed");
  }
  tokens.resize(n_tokens);
  return tokens;
}

std::pair<double, int> BenchmarkWrapper::run_base_inference(){
  try{
    std::print("Starting base inference...\n");
    auto start = std::chrono::high_resolution_clock::now();

    const char* prompt = "Tell me about machine learning.";

    std::print("Tokenizing input...\n");
    std::vector<llama_token> tokens = tokenize_input(model, prompt);
    std::print("Tokenized {} tokens\n", tokens.size());
    
    std::print("Creating batch...\n");
    // Create batch for processing
    llama_batch batch = llama_batch_init(
      (int)tokens.size(),
      0,
      4
    );

    // Fill the batch
    for(int i = 0; i < (int)tokens.size(); ++i){
      batch.token[i] = tokens[i];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = (i == (int) tokens.size() - 1) ? 1 : 0;
    }

    batch.n_tokens = tokens.size();
    std::print("Processing initial batch...\n");
    // Process batch
    if(llama_decode(ctx, batch) != 0){ 
      llama_batch_free(batch);
      throw std::runtime_error("Failed to decode"); 
    }

    const int n_max_tokens = 512;
    int tokens_generated = 0;
    int last_token_pos = tokens.size()-1;
    
    std::print("Starting token generation...\n");
    for(int i = 0; i < n_max_tokens; i++){
      std::print("Sampling token {} at position {}\n", i, last_token_pos);
      int sampling_pos = (i == 0)? (tokens.size() - 1): 0;
      llama_token new_token = llama_sampler_sample(sampler, ctx, sampling_pos);
      std::print("Sampled token: {}\n", new_token);
      if(new_token == llama_token_eos(model)){
        std::print("Reached EOS token\n");
        break;
      }
      tokens_generated++;
      llama_batch new_batch = llama_batch_init(
        1, // n_token
        0,
        4
      );

      new_batch.token[0] = new_token;
      new_batch.pos[0] = last_token_pos;
      new_batch.n_seq_id[0] = 1;
      new_batch.seq_id[0][0] = 0;
      new_batch.logits[0] = 1;
      new_batch.n_tokens = 1;

      std::print("Decoding token...\n");
      if(llama_decode(ctx, new_batch) != 0){
        llama_batch_free(new_batch);
        llama_batch_free(batch);
        throw std::runtime_error("Failed to decode");
      }
      llama_batch_free(new_batch);
    }
    llama_batch_free(batch);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::print("WE GOT HERE");
    return {diff.count(), tokens_generated};
  } catch(const std::exception& e){
    std::print(stderr, "Error running base inference: {}\n", e.what());
    throw;
  }
}

std::pair<double, int> BenchmarkWrapper::run_optimized_inference(){
  auto start = std::chrono::high_resolution_clock::now();
  
  const int BATCH_SIZE = 512;
  // Pre-allolcate batch
  llama_batch batch = llama_batch_init(
    /*n_tokens=*/  BATCH_SIZE,
    /*embd=*/       0,
    /*n_seq_max=*/  4
  );
  // Initial tokenization
  std::vector<llama_token> tokens;
  tokens.resize(BATCH_SIZE);

  const char* prompt = "Tell me about machine learning";
  int n_tokens = llama_tokenize(model, prompt, -1, tokens.data(), BATCH_SIZE, true, false);
  tokens.resize(n_tokens);
  
  // PROCESSING PROMPT
  // Fill batch with tokens
  for(int i = 0; i < n_tokens; ++i){
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = 1;
  }
  batch.n_tokens = n_tokens;
  // Processing entire batch at once
  if(llama_decode(ctx, batch) != 0) { throw std::runtime_error("Failed to decode batch."); }

  const int n_max_tokens = 512;
  int tokens_generated = 0;

  while(tokens_generated < n_max_tokens){
    // Fill batch with new tokens
    batch.n_tokens = 0;
    while(batch.n_tokens < BATCH_SIZE && tokens_generated < n_max_tokens){
      llama_token new_token = llama_sampler_sample(sampler, ctx, tokens_generated);
      if(new_token == llama_token_eos(model)){break;}
      
      int pos = batch.n_tokens;
      batch.token[pos] = new_token;
      batch.pos[pos] = tokens_generated + n_tokens;
      batch.n_seq_id[pos] = 1;
      batch.seq_id[pos][0] = 0;
      batch.logits[pos] = 1;

      batch.n_tokens++;
      tokens_generated++;
    }
    if(batch.n_tokens == 0){break;}
    if(llama_decode(ctx, batch) != 0 ){ throw std::runtime_error("Failed to decode batch"); }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::print("WE GOT HERE");
  return {diff.count(), tokens_generated};

}

void BenchmarkWrapper::runner(const char* run_type){
  std::pair<double, int> result{};
  if(strcmp(run_type, "base") == 0){
    result = run_base_inference();
  }else if(strcmp(run_type, "optimized") == 0){
    result = run_optimized_inference();
  }else{
    std::print(stderr, "Failed to run inference");
  }
  std::println("{0},{1}", result.first, result.second);
}

BenchmarkWrapper::~BenchmarkWrapper(){
  if(sampler){llama_sampler_free(sampler);}
  llama_free(ctx);
  llama_free_model(model);
}

