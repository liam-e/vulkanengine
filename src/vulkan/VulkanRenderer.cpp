#include "VulkanRenderer.h"

VulkanTriangle::VulkanTriangle() {
    queueIndices.graphicsFamily = UINT32_MAX;
    queueIndices.presentFamily = UINT32_MAX;
    physicalDevice = VK_NULL_HANDLE;
    currentFrame = 0;
}

void VulkanTriangle::Run() {
    InitWindow();
    InitVulkan();
    MainLoop();
    Cleanup();
}

void VulkanTriangle::InitWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(1280, 720, "Vulkan Triangle", nullptr, nullptr);
}

void VulkanTriangle::InitVulkan() {
    CreateInstance();
    CreateSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapchain();
    imagesInFlight.resize(swapchainImages.size(), VK_NULL_HANDLE);
    CreateImageViews();
    CreateRenderPass();
    CreateDescriptorSetLayout();
    CreateVertexBuffer();
    CreateIndexBuffer();
    CreateUniformBuffer();
    CreateDescriptorPool();
    CreateDescriptorSet();
    CreateGraphicsPipeline();
    CreateFramebuffers();
    CreateCommandPool();
    CreateCommandBuffers();
    CreateSyncObjects();
}

void VulkanTriangle::MainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        DrawFrame();
    }
    vkDeviceWaitIdle(device);
}

void VulkanTriangle::Cleanup() {
    vkDeviceWaitIdle(device);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    for (size_t i = 0; i < renderFinishedSemaphores.size(); ++i) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);
    for (auto framebuffer : swapchainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    for (auto imageView : swapchainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);

    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, uniformBuffer, nullptr);
    vkUnmapMemory(device, uniformBufferMemory);
    vkFreeMemory(device, uniformBufferMemory, nullptr);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}