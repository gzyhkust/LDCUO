#pragma once

// Save the assert definition before including PyTorch
#include <cassert>

// Standard library headers
#include <vector>
#include <cmath>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <string>

// Include PyTorch headers (may change macros)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <torch/torch.h>
#include <torch/script.h>
#pragma GCC diagnostic pop

namespace hnswlib {

/**
 * PyTorch SiameseNetwork distance calculator
 * Used to compute distance between two vectors in learned embedding space
 */
class PyTorchDistanceCalculator {
private:
    torch::jit::script::Module model_;
    bool model_loaded_;
    int input_dim_;
    int embed_dim_;
    torch::Device device_;

public:
    PyTorchDistanceCalculator(const std::string& model_path, 
                             int input_dim = 960, 
                             int embed_dim = 32,
                             bool use_cuda = false)
        : input_dim_(input_dim), embed_dim_(embed_dim), model_loaded_(false),
          device_(use_cuda ? torch::kCUDA : torch::kCPU) {
        try {
            // Load TorchScript model
            model_ = torch::jit::load(model_path, device_);
            model_.eval();  // Set to evaluation mode
            model_loaded_ = true;
            std::cout << "Successfully loaded PyTorch model from: " << model_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            model_loaded_ = false;
        }
    }

    /**
     * Compute distance between two vectors
     * @param vec1 first vector pointer (float*)
     * @param vec2 second vector pointer (float*)
     * @param dim vector dimension
     * @return Euclidean distance
     */
    float computeDistance(const void* vec1, const void* vec2, int dim) {
        if (!model_loaded_) {
            throw std::runtime_error("PyTorch model not loaded");
        }

        try {
            // Convert pointers to float arrays
            const float* arr1 = static_cast<const float*>(vec1);
            const float* arr2 = static_cast<const float*>(vec2);

            // Validate dimension
            if (dim != input_dim_) {
                throw std::runtime_error("Input dimension mismatch: expected " + 
                                       std::to_string(input_dim_) + ", got " + 
                                       std::to_string(dim));
            }

            // Create PyTorch tensors (from vector data)
            // Use unsqueeze(0) to add batch dimension: (dim,) -> (1, dim)
            torch::Tensor tensor1 = torch::from_blob(
                (void*)arr1, {dim}, torch::kFloat32
            ).clone().unsqueeze(0).to(device_);
            
            torch::Tensor tensor2 = torch::from_blob(
                (void*)arr2, {dim}, torch::kFloat32
            ).clone().unsqueeze(0).to(device_);

            // Disable gradient computation (inference mode)
            torch::NoGradGuard no_grad;

            // Get embeddings through model (SiameseNetwork needs two inputs)
            // Model forward(input1, input2) returns (embed1, embed2)
            std::vector<torch::IValue> inputs{tensor1, tensor2};
            auto outputs = model_.forward(inputs);
            
            // Unpack tuple to get two embedding vectors
            torch::Tensor embed1 = outputs.toTuple()->elements()[0].toTensor();
            torch::Tensor embed2 = outputs.toTuple()->elements()[1].toTensor();

            // Compute Euclidean distance: L2 norm of difference
            // embed1, embed2 shape: (1, embed_dim)
            torch::Tensor diff = embed1 - embed2;
            
            // Use norm to compute L2 distance, on dim=1
            torch::Tensor dist_tensor = torch::norm(diff, /*p=*/2, /*dim=*/1);
            
            // Extract scalar value and move to CPU
            float distance = dist_tensor.item<float>();
            
            return distance;

        } catch (const std::exception& e) {
            std::cerr << "Error computing distance: " << e.what() << std::endl;
            throw;
        }
    }

    /**
     * Check if model is successfully loaded
     */
    bool isModelLoaded() const {
        return model_loaded_;
    }

    /**
     * Get input dimension
     */
    int getInputDim() const {
        return input_dim_;
    }

    /**
     * Get embedding dimension
     */
    int getEmbedDim() const {
        return embed_dim_;
    }

    /**
     * Get device (CPU or CUDA)
     */
    torch::Device getDevice() const {
        return device_;
    }

    /**
     * Directly call model forward method
     * Used for batch embedding operations
     * @param inputs input tensor list
     * @return model output
     */
    torch::IValue forward(const std::vector<torch::IValue>& inputs) {
        if (!model_loaded_) {
            throw std::runtime_error("PyTorch model not loaded");
        }
        return model_.forward(inputs);
    }
};

}  // namespace hnswlib