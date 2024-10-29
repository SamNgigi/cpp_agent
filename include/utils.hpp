#ifndef _UTILS_HPP
#define _UTILS_HPP

#include <string>
#include <unordered_map>


namespace llm_agent {
  namespace utils {
    std::string get_project_root();
    std::unordered_map<std::string, std::string> read_env(const std::string& relative_path=".env");
  }
}

#endif //UTILS_HPP
