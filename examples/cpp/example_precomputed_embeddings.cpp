#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "hnswlib/hnswlib.h"

/**
 * Example: vector search using precomputed embeddings
 * 
 * Workflow:
 * 1. Create HNSW index and add base vectors
 * 2. Initialize PyTorch model
 * 3. Call embedAllVectors() to embed all vectors
 * 4. Embed query vectors
 * 5. Execute search in embedding space
 */

void printUsage() {
    std::cout << "\nUsage:" << std::endl;
    std::cout << "  ./example_precomputed_embeddings <num_vectors> <dim> <model_path>" << std::endl;
    std::cout << "\nParameter description:" << std::endl;
    std::cout << "  num_vectors  - number of vectors to add to index" << std::endl;
    std::cout << "  dim          - vector dimension (default 960)" << std::endl;
    std::cout << "  model_path   - TorchScript model file path (siamese_model.pt)" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  ./example_precomputed_embeddings 1000 960 ./siamese_model.pt" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Precomputed embeddings vector search example" << std::endl;
        std::cout << "==============================" << std::endl;
        printUsage();
        return 1;
    }

    // ============ Parameter settings ============
    int num_vectors = std::stoi(argv[1]);
    int dim = 960;  // vector dimension
    std::string model_path = "siamese_model.pt";

    if (argc >= 3) {
        dim = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        model_path = argv[3];
    }

    std::cout << "\n========== Configuration ==========" << std::endl;
    std::cout << "Number of vectors: " << num_vectors << std::endl;
    std::cout << "Vector dimension: " << dim << std::endl;
    std::cout << "Model path: " << model_path << std::endl;

    // ============ Step 1: Create HNSW index ============
    std::cout << "\n[Step 1] Creating HNSW index..." << std::endl;

    int M = 16;  // HNSW parameter
    int ef_construction = 200;
    int max_elements = num_vectors;

    // Create L2 space
    hnswlib::L2Space space(dim);

    // Create HNSW index
    hnswlib::HierarchicalNSW<float> hnsw_index(&space, max_elements, M, ef_construction);

    std::cout << "✓ HNSW index created" << std::endl;
    std::cout << "  M: " << M << ", ef_construction: " << ef_construction << std::endl;

    // ============ Step 2: Generate random vectors and add to index ============
    std::cout << "\n[Step 2] Adding vectors to index..." << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    std::vector<std::vector<float>> base_vectors;
    
    for (int i = 0; i < num_vectors; i++) {
        std::vector<float> vec(dim);
        for (int j = 0; j < dim; j++) {
            vec[j] = dis(gen);
        }
        base_vectors.push_back(vec);
        hnsw_index.addPoint(vec.data(), i);
        
        if ((i + 1) % 200 == 0) {
            std::cout << "  Added " << (i + 1) << "/" << num_vectors << " vectors" << std::endl;
        }
    }

    std::cout << "✓ Added " << num_vectors << " vectors to index" << std::endl;

    // ============ Step 3: Initialize PyTorch model ============
    std::cout << "\n[Step 3] Initializing PyTorch model..." << std::endl;

    hnsw_index.initPyTorchDistance(model_path, false);  // false = use CPU
    std::cout << "✓ PyTorch model initialized" << std::endl;

    // ============ Step 4: Batch embed all vectors ============
    std::cout << "\n[Step 4] Batch embedding all vectors..." << std::endl;
    std::cout << "This step converts all base vectors to low-dimensional embeddings" << std::endl;

    auto start_embed = std::chrono::high_resolution_clock::now();
    
    bool embed_success = hnsw_index.embedAllVectors();
    
    auto end_embed = std::chrono::high_resolution_clock::now();
    auto embed_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_embed - start_embed);

    if (!embed_success) {
        std::cerr << "❌ embedding failed" << std::endl;
        return 1;
    }

    std::cout << "✓ embedding complete, time: " << embed_duration.count() << " ms" << std::endl;

    // ============ Step 5: Save embeddings to file (optional) ============
    std::cout << "\n[Step 5] Saving embeddings to file (optional)..." << std::endl;
    
    std::string embeddings_file = "embeddings.bin";
    if (hnsw_index.saveEmbeddings(embeddings_file)) {
        std::cout << "✓ embeddings saved to " << embeddings_file << std::endl;
    }

    // ============ Step 6: Execute search ============
    std::cout << "\n[Step 6] Executing k-NN search..." << std::endl;

    int num_queries = 10;
    int k = 10;

    hnsw_index.setEf(100);  // Set search parameter ef

    for (int q = 0; q < num_queries; q++) {
        // Generate random query vector
        std::vector<float> query_vec(dim);
        for (int j = 0; j < dim; j++) {
            query_vec[j] = dis(gen);
        }

        std::cout << "\nQuery " << (q + 1) << ": " << std::endl;

        // Execute search
        auto start_search = std::chrono::high_resolution_clock::now();
        
        auto result = hnsw_index.searchKnn(query_vec.data(), k);
        
        auto end_search = std::chrono::high_resolution_clock::now();
        auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_search - start_search);

        // Output results
        std::cout << "  k=" << k << ", search time: " << search_duration.count() << " μs" << std::endl;
        std::cout << "  Nearest neighbor results:" << std::endl;

        int rank = 1;
        while (!result.empty()) {
            auto item = result.top();
            result.pop();
            std::cout << "    " << rank << ". Label: " << item.second 
                      << ", Distance: " << std::fixed << std::setprecision(6) << item.first << std::endl;
            rank++;
        }
    }

    // ============ Step 7: Performance comparison (optional) ============
    std::cout << "\n[Step 7] Performance comparison..." << std::endl;
    std::cout << "\nDisabling precomputed embeddings mode, using original vectors for distance calculation..." << std::endl;
    
    hnsw_index.disablePrecomputedEmbeddings();

    int comparison_queries = 3;
    
    std::cout << "\nExecuting " << comparison_queries << " queries for comparison..." << std::endl;

    double total_time_with_embedding = 0;
    double total_time_without_embedding = 0;

    for (int q = 0; q < comparison_queries; q++) {
        // Generate random query vector
        std::vector<float> query_vec(dim);
        for (int j = 0; j < dim; j++) {
            query_vec[j] = dis(gen);
        }

        // Use original vectors for distance calculation
        auto start = std::chrono::high_resolution_clock::now();
        auto result = hnsw_index.searchKnn(query_vec.data(), k);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time_without_embedding += duration.count();

        std::cout << "  Query " << (q + 1) << " (original vectors): " << duration.count() << " μs" << std::endl;
    }

    std::cout << "\nAverage search time (original vectors): " 
              << total_time_without_embedding / comparison_queries << " μs" << std::endl;

    // Re-enable precomputed embeddings
    std::cout << "\nRe-enabling precomputed embeddings mode..." << std::endl;
    if (hnsw_index.loadEmbeddings(embeddings_file)) {
        std::cout << "✓ embeddings loaded" << std::endl;

        std::cout << "\nExecuting " << comparison_queries << " queries for comparison..." << std::endl;

        for (int q = 0; q < comparison_queries; q++) {
            // Generate random query vector
            std::vector<float> query_vec(dim);
            for (int j = 0; j < dim; j++) {
                query_vec[j] = dis(gen);
            }

            // Use embeddings for distance calculation
            auto start = std::chrono::high_resolution_clock::now();
            auto result = hnsw_index.searchKnn(query_vec.data(), k);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            total_time_with_embedding += duration.count();

            std::cout << "  Query " << (q + 1) << " (embeddings): " << duration.count() << " μs" << std::endl;
        }

        std::cout << "\nAverage search time (embeddings): " 
                  << total_time_with_embedding / comparison_queries << " μs" << std::endl;

        double speedup = total_time_without_embedding / total_time_with_embedding;
        std::cout << "\nPerformance speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }

    std::cout << "\n========== Test completed ==========" << std::endl;
    return 0;
}
