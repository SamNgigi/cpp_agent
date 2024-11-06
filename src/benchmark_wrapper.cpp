#include "benchmark_wrapper.hpp"
#include <llama.h>

#include <chrono>
#include <print>


BenchmarkWrapper::BenchmarkWrapper(const char* model_path){
  // Initialize model
  std::print(stderr, "Initializing model from path: {}\n\n", model_path);
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0; // CPU only for testing
  model_params.vocab_only = false;
  model = llama_load_model_from_file(model_path, model_params);
  if(!model){ throw std::runtime_error("Failed to load model"); }
  std::print(stderr, "\nModel loaded successfully\n\n");

  
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
  std::print(stderr, "\nContext loaded successfully\n");

  // Intialize sampler
  sampler = llama_sampler_init_greedy();
  if(!sampler){
    llama_free(ctx);
    llama_free_model(model);
    throw std::runtime_error("Failed to initialize sampler");
  }
  std::print(stderr, "\nSampler initialized successfully\n");
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
    //std::print(stderr, "Starting base inference...\n");
    auto start = std::chrono::high_resolution_clock::now();

    const char* prompt = "Tell me about machine learning.";
    //std::print(stderr, "Tokenizing input...\n");
    std::vector<llama_token> tokens = tokenize_input(model, prompt);
    //std::print(stderr, "Tokenized {} tokens\n", tokens.size());
    
    //std::print(stderr, "Creating batch...\n");
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
    //std::print(stderr, "Processing initial batch...\n");
    // Process batch
    if(llama_decode(ctx, batch) != 0){ 
      llama_batch_free(batch);
      throw std::runtime_error("Failed to decode"); 
    }

    const int n_max_tokens = 512;
    int tokens_generated = 0;
    int last_logits_pos= 0;
    
    //std::print(stderr, "Starting token generation...\n");
    for(int i = 0; i < n_max_tokens; i++){
      //std::print(stderr, "Sampling token {} at position {}\n", i, last_logits_pos);
      int sampling_pos = (i == 0)? (tokens.size() - 1): 0;
      llama_token new_token = llama_sampler_sample(sampler, ctx, sampling_pos);
      //std::print(stderr, "Sampled token: {}\n", new_token);
      if(new_token == llama_token_eos(model)){
        //std::print(stderr, "Reached EOS token\n");
        break;
      }
      tokens_generated++;
      llama_batch new_batch = llama_batch_init(
        1, // n_token
        0,
        4
      );

      new_batch.token[0] = new_token;
      new_batch.pos[0] = 0; // Set position within batch
      new_batch.n_seq_id[0] = 1;
      new_batch.seq_id[0][0] = 0;
      new_batch.logits[0] = 1;
      new_batch.n_tokens = 1;

      // std::print(stderr, "Decoding token...\n");
      if(llama_decode(ctx, new_batch) != 0){
        llama_batch_free(new_batch);
        llama_batch_free(batch);
        throw std::runtime_error("Failed to decode");
      }
      llama_batch_free(new_batch);
      // Update last_logits_pos to the position within the batch
      last_logits_pos = 0; // Since new_batch.pos[0] = 0;
    }
    llama_batch_free(batch);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return {diff.count(), tokens_generated};
  } catch(const std::exception& e){
    std::print(stderr, "Error running base inference: {}\n", e.what());
    throw;
  }
}

std::pair<double, int> BenchmarkWrapper::run_optimized_inference() {
  try {
      std::print(stderr, "Starting optimized inference...\n");
      auto start = std::chrono::high_resolution_clock::now();

      const char* prompt = "Tell me about machine learning.";
      std::vector<llama_token> tokens = tokenize_input(model, prompt);
      std::print(stderr, "Tokenized {} tokens\n", tokens.size());

      // Constants for optimized batching
      const int BATCH_SIZE = 32;  // Process multiple tokens at once
      const int n_max_tokens = 512;
      
      // Pre-allocate a larger batch for efficiency
      llama_batch batch = llama_batch_init(
          BATCH_SIZE,  // n_tokens - larger batch size
          0,          // embd
          4           // n_seq_max
      );

      // First, process the prompt
      std::print(stderr, "Processing prompt...\n");
      int current_pos = 0;
      while (current_pos < tokens.size()) {
          int tokens_remaining = tokens.size() - current_pos;
          int tokens_to_process = std::min(BATCH_SIZE, tokens_remaining);
          
          // Fill batch with prompt tokens
          for (int i = 0; i < tokens_to_process; i++) {
              batch.token[i] = tokens[current_pos + i];
              batch.pos[i] = i;
              batch.n_seq_id[i] = 1;
              batch.seq_id[i][0] = 0;
              batch.logits[i] = (i == tokens_to_process - 1) ? 1 : 0;
          }
          batch.n_tokens = tokens_to_process;

          std::print(stderr, "Processing prompt batch of {} tokens at position {}\n", 
                    tokens_to_process, current_pos);

          if (llama_decode(ctx, batch) != 0) {
              llama_batch_free(batch);
              throw std::runtime_error("Failed to decode prompt batch");
          }
          
          current_pos += tokens_to_process;
      }

      int tokens_generated = 0;
      int sequence_pos = tokens.size();
      std::vector<llama_token> generated_tokens;
      int last_logits_pos = tokens.size() - 1;  // Track position of last computed logits
      
      std::print(stderr, "Starting optimized token generation from position {}...\n", sequence_pos);
      
      while (tokens_generated < n_max_tokens) {
          // Sample next token using last valid logits position
          std::print(stderr, "Sampling using logits from position {}\n", last_logits_pos);
          llama_token new_token = llama_sampler_sample(sampler, ctx, last_logits_pos);
          
          if (new_token == llama_token_eos(model)) {
              std::print(stderr, "Reached EOS token\n");
              break;
          }
          
          generated_tokens.push_back(new_token);
          tokens_generated++;

          // Process batch when we have enough tokens or hit special conditions
          if (generated_tokens.size() == BATCH_SIZE || 
              tokens_generated == n_max_tokens || 
              new_token == llama_token_eos(model)) {
              
              std::print(stderr, "Processing generation batch of {} tokens at position {}\n", 
                        generated_tokens.size(), sequence_pos);

              // Fill the batch with accumulated tokens
              for (size_t i = 0; i < generated_tokens.size(); i++) {
                  batch.token[i] = generated_tokens[i];
                  batch.pos[i] =   i;
                  batch.n_seq_id[i] = 1;
                  batch.seq_id[i][0] = 0;
                  batch.logits[i] = (i == generated_tokens.size() - 1) ? 1 : 0;
              }
              batch.n_tokens = generated_tokens.size();
              // Update last logits position to the last token in this batch
              last_logits_pos = batch.n_tokens - 1;

              // Process the batch
              if (llama_decode(ctx, batch) != 0) {
                  llama_batch_free(batch);
                  throw std::runtime_error("Failed to decode generation batch");
              }


              sequence_pos += generated_tokens.size();
              generated_tokens.clear();
          }
      }

      llama_batch_free(batch);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;
      
      std::print(stderr, "Optimized inference completed. Generated {} tokens\n", tokens_generated);
      return {diff.count(), tokens_generated};
      
  } catch(const std::exception& e) {
      std::print(stderr, "Error running optimized inference: {}\n", e.what());
      throw;
  }
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

