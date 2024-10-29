#include <print>
#include <llama.h>


int main(){

  std::println("Hello World!");
  llama_backend_init();
  // llama.cpp code here
  llama_backend_free();
  return EXIT_SUCCESS;
}
