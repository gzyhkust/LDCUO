#pragma once

#include <vector>
#include <memory>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <fstream>

namespace hnswlib {

/**
 * Embeddings Manager: for storing and managing precomputed vector embeddings
 * 
 * Design concept:
 * - Store embeddings of all vectors (instead of raw vectors)
 * - Use ID as index for fast access
 * - Support dynamic expansion
 */
class EmbeddingsManager {
private:
    // embeddings data storage: using char* to store float array
    // Similar to HNSW's data_level0_memory_ approach
    char* embeddings_data_{nullptr};
    
    // embedding dimension
    uint32_t embedding_dim_{32};
    
    // current number of embeddings stored
    uint32_t num_embeddings_{0};
    
    // maximum embeddings capacity
    uint32_t max_embeddings_{0};
    
    // bytes per embedding
    size_t bytes_per_embedding_{0};

public:
    /**
     * Constructor
     * @param max_embeddings maximum number of embeddings
     * @param embedding_dim embedding dimension
     */
    EmbeddingsManager(uint32_t max_embeddings = 100000, uint32_t embedding_dim = 32)
        : max_embeddings_(max_embeddings), embedding_dim_(embedding_dim),
          num_embeddings_(0) {
        
        bytes_per_embedding_ = embedding_dim * sizeof(float);
        
        // Allocate memory: similar to HNSW's data_level0_memory_
        embeddings_data_ = (char*)malloc(max_embeddings * bytes_per_embedding_);
        if (embeddings_data_ == nullptr) {
            throw std::runtime_error("EmbeddingsManager: Failed to allocate memory for embeddings");
        }
    }

    /**
     * Destructor
     */
    ~EmbeddingsManager() {
        if (embeddings_data_ != nullptr) {
            free(embeddings_data_);
            embeddings_data_ = nullptr;
        }
    }

    /**
     * Disable copy constructor
     */
    EmbeddingsManager(const EmbeddingsManager&) = delete;
    EmbeddingsManager& operator=(const EmbeddingsManager&) = delete;

    /**
     * Add or update an embedding
     * @param id embedding ID
     * @param embedding embedding data pointer
     */
    void addEmbedding(uint32_t id, const float* embedding) {
        if (id >= max_embeddings_) {
            throw std::runtime_error("EmbeddingsManager: ID exceeds max capacity");
        }

        // Copy embedding to memory
        char* dest = embeddings_data_ + id * bytes_per_embedding_;
        memcpy(dest, embedding, bytes_per_embedding_);
        
        // Update num_embeddings
        if (id >= num_embeddings_) {
            num_embeddings_ = id + 1;
        }
    }

    /**
     * Get embedding pointer for specified ID
     * @param id embedding ID
     * @return embedding pointer (float*)
     */
    const float* getEmbedding(uint32_t id) const {
        if (id >= num_embeddings_) {
            return nullptr;
        }
        return reinterpret_cast<const float*>(
            embeddings_data_ + id * bytes_per_embedding_
        );
    }

    /**
     * Get writable embedding pointer (for direct writing)
     * @param id embedding ID
     * @return embedding pointer (float*)
     */
    float* getEmbeddingMutable(uint32_t id) {
        if (id >= max_embeddings_) {
            return nullptr;
        }
        if (id >= num_embeddings_) {
            num_embeddings_ = id + 1;
        }
        return reinterpret_cast<float*>(
            embeddings_data_ + id * bytes_per_embedding_
        );
    }

    /**
     * Expand capacity
     * @param new_max_embeddings new maximum capacity
     */
    void resize(uint32_t new_max_embeddings) {
        if (new_max_embeddings <= max_embeddings_) {
            return;
        }

        char* new_data = (char*)realloc(
            embeddings_data_,
            new_max_embeddings * bytes_per_embedding_
        );
        
        if (new_data == nullptr) {
            throw std::runtime_error("EmbeddingsManager: Failed to resize embeddings memory");
        }

        embeddings_data_ = new_data;
        max_embeddings_ = new_max_embeddings;
    }

    /**
     * Get current number of embeddings
     */
    uint32_t getNumEmbeddings() const {
        return num_embeddings_;
    }

    /**
     * Get embedding dimension
     */
    uint32_t getEmbeddingDim() const {
        return embedding_dim_;
    }

    /**
     * Get maximum capacity
     */
    uint32_t getMaxEmbeddings() const {
        return max_embeddings_;
    }

    /**
     * Clear all embeddings
     */
    void clear() {
        num_embeddings_ = 0;
    }

    /**
     * Compute L2 distance between two embeddings
     * @param emb1 first embedding pointer
     * @param emb2 second embedding pointer
     * @return L2 distance
     */
    static float computeEmbeddingDistance(const float* emb1, const float* emb2, uint32_t dim) {
        if (!emb1 || !emb2) {
            return std::numeric_limits<float>::max();
        }

        float dist = 0;
        for (uint32_t i = 0; i < dim; i++) {
            float diff = emb1[i] - emb2[i];
            dist += diff * diff;
        }
        return sqrt(dist);
    }

    /**
     * Save embeddings to file
     * @param filename file path
     */
    bool saveToFile(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "EmbeddingsManager: Failed to open file: " << filename << std::endl;
            return false;
        }

        // Write header info
        file.write(reinterpret_cast<const char*>(&embedding_dim_), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&num_embeddings_), sizeof(uint32_t));
        
        // Write embeddings data
        file.write(embeddings_data_, num_embeddings_ * bytes_per_embedding_);
        
        file.close();
        std::cout << "EmbeddingsManager: Saved " << num_embeddings_ << " embeddings to " 
                  << filename << std::endl;
        return true;
    }

    /**
     * Load embeddings from file
     * @param filename file path
     */
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "EmbeddingsManager: Failed to open file: " << filename << std::endl;
            return false;
        }

        // Read header info
        uint32_t loaded_dim, loaded_count;
        file.read(reinterpret_cast<char*>(&loaded_dim), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&loaded_count), sizeof(uint32_t));
        
        // Check dimension match
        if (loaded_dim != embedding_dim_) {
            std::cerr << "EmbeddingsManager: Dimension mismatch. Expected " << embedding_dim_ 
                      << ", got " << loaded_dim << std::endl;
            return false;
        }

        // Check capacity
        if (loaded_count > max_embeddings_) {
            resize(loaded_count);
        }

        // Read embeddings data
        file.read(embeddings_data_, loaded_count * bytes_per_embedding_);
        num_embeddings_ = loaded_count;

        file.close();
        std::cout << "EmbeddingsManager: Loaded " << num_embeddings_ << " embeddings from " 
                  << filename << std::endl;
        return true;
    }

    /**
     * Get raw data pointer (for debugging or special operations)
     */
    char* getRawDataPtr() {
        return embeddings_data_;
    }

    const char* getRawDataPtr() const {
        return embeddings_data_;
    }

    /**
     * Get bytes per embedding
     */
    size_t getBytesPerEmbedding() const {
        return bytes_per_embedding_;
    }
};

}  // namespace hnswlib
