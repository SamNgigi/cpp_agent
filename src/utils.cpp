#include "utils.hpp"

#include <print>
#include <filesystem>
#include <fstream>
#include <iostream>


namespace fs = std::filesystem;

    
    
std::string llm_agent::utils::get_project_root(){
  
  fs::path exe_path = fs::current_path();

  if(exe_path.empty()){
    throw std::runtime_error("Could not find project root (no .env file in parent directories)");
  }
  return exe_path.string();

}

std::unordered_map<std::string, std::string> llm_agent::utils::read_env(const std::string& relative_path){
  std::unordered_map<std::string, std::string> env{};
  
  fs::path env_path = fs::path(llm_agent::utils::get_project_root()) / relative_path;

  
  std::ifstream file(env_path);
  if(!file.is_open()){
    std::cerr << "Warning: Could not open" << env_path << "\n";
    return env;
  }

  std::string line;
  while(std::getline(file, line)){

    if(line.empty() || line[0] == '#') {continue;}
    
    auto sep = line.find('=');
    if(sep != std::string::npos){
      env[line.substr(0, sep)] = line.substr(sep + 1);
    }
  }

  return env;
  
}


