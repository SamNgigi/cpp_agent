#include "utils.hpp"

#include <print>
#include <iostream>
#include <string>
#include <llama.h>


int main(){

  std::print("Hello World!!\n\n");
  auto res = llama_print_system_info();
  std::print("llama sytem info::\n{}\n\n", res);
  std::unordered_map<std::string, std::string> env = llm_agent::utils::read_env(); 
  std::print("model_path :: {}\n", env["CODE_LLAMA"]);
  return EXIT_SUCCESS;
}
