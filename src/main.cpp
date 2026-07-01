#include <iostream>
#include <string>
#include "tui/app.hpp"

int main() {
    std::string prompt;
    std::cout << "Enter a prompt to trace through the model: ";
    std::getline(std::cin, prompt);
    if (prompt.empty()) {
        prompt = "The capital of France is";
    }

    TracerApp app;
    app.run(prompt);
    return 0;
}