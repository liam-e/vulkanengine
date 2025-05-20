#include "VulkanRenderer.h"
#include <stdexcept>
#include <iostream>

int main() {
    VulkanTriangle app;
    try {
        app.Run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}