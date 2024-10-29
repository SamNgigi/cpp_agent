#include "utils.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <llama.h>
#include <filesystem>


namespace fs = std::filesystem;


class LlamaCppSetupTest : public ::testing::Test{
protected:
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  std::unordered_map<std::string, std::string> env = llm_agent::utils::read_env();
  const std::string model_path = env["CODE_LLAMA"];

  void SetUp() override {
    if(fs::exists(model_path)){
      llama_model_params mdl_params = llama_model_default_params();
      mdl_params.n_gpu_layers = 0; // CPU only for testing
      
      llama_context_params ctx_params = llama_context_default_params();
      ctx_params.n_ctx = 512; // Smaller context for testing
      ctx_params.n_batch = 512;

      model = llama_load_model_from_file(model_path.c_str(), mdl_params);
      ASSERT_NE(model, nullptr) << "Failed to load model from " << model_path;
      ctx = llama_new_context_with_model(model, ctx_params);
      ASSERT_NE(ctx, nullptr) << "Failed to create context";

    }
  }

  void TearDown() override {
    if(ctx != nullptr){
      llama_free(ctx);
    }
    if(model != nullptr){
      llama_free_model(model);
    }
  }

  bool IsModelAvailable() const {
    return fs::exists(model_path);
  }
};


TEST_F(LlamaCppSetupTest, TestModelFileExists){
  ASSERT_TRUE(IsModelAvailable()) << "Model file not found at" << model_path;
}

TEST_F(LlamaCppSetupTest, TestBasicInference){
  if(!IsModelAvailable()) {GTEST_SKIP();}

  std::string prompt = "Print 'Hello World' in Python";
  std::vector<llama_token> tokens(32);

  // Tokenize input
  
  int n_tokens = llama_tokenize(model, 
                                prompt.data(), 
                                prompt.length(),
                                tokens.data(),
                                tokens.size(),
                                true,
                                false);
  ASSERT_GT(n_tokens, 0) << "Failed to tokenize input prompt";
  
  // Create and process batch
  llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
  ASSERT_EQ(llama_decode(ctx, batch), 0) << "Failed to decode batch";

  // Get logits for next token
  float* logits = llama_get_logits(ctx);
  ASSERT_NE(logits, nullptr) << "Failed to get logits";

  // Test the presence of embeddings if enabled
  if(llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE){
    float* embeddings = llama_get_embeddings(ctx);
    EXPECT_NE(embeddings, nullptr) << "Failed to get embeddings";
  }

}
