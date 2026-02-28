#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include "hnswlib/hnswlib.h"


// Struct: neighbors classified by hop distance
struct HopRangeNeighbors {
    std::vector<int> hop1;      // neighbors within 1 hop
    std::vector<int> hop2;     // neighbors within 2 hops
    std::vector<int> hop3;      // neighbors beyond 2 hops
};

/**
 * Find neighbors by hop distance in the layer 0 graph structure
 * @param alg_hnsw HNSW index pointer
 * @param start_id starting node id
 * @return HopRangeNeighbors struct containing neighbors classified by hop distance
 */
HopRangeNeighbors findNeighborsByHopRange(hnswlib::HierarchicalNSW<float>* alg_hnsw, int start_id) {
    HopRangeNeighbors result;
    
    // Use BFS to traverse the graph
    std::queue<std::pair<int, int>> bfs_queue;  // (node_id, hop_count)
    std::unordered_set<int> visited;
    
    // Initialize: add starting node to queue
    bfs_queue.push({start_id, 0});
    visited.insert(start_id);
    
    while (!bfs_queue.empty()) {
        auto curr_pair = bfs_queue.front();
        bfs_queue.pop();
        int curr_id = curr_pair.first;
        int hop_count = curr_pair.second;
        
        // Get layer 0 link list
        hnswlib::linklistsizeint *linklist = alg_hnsw->get_linklist0(curr_id);
        size_t neighbor_count = alg_hnsw->getListCount(linklist);
        hnswlib::tableint *neighbors = (hnswlib::tableint*)(linklist + 1);
        
        // Traverse all adjacent nodes
        for (size_t i = 0; i < neighbor_count; i++) {
            int neighbor_id = neighbors[i];
            
            // Check if already visited
            if (visited.find(neighbor_id) == visited.end()) {
                visited.insert(neighbor_id);
                
                int next_hop = hop_count + 1;
                
                // Classify neighbors by hop distance
                if (next_hop <= 1) {
                    result.hop1.push_back(neighbor_id);
                    bfs_queue.push({neighbor_id, next_hop});
                } else if (next_hop <= 2) {
                    result.hop2.push_back(neighbor_id);
                    bfs_queue.push({neighbor_id, next_hop});
                } else {
                    result.hop3.push_back(neighbor_id);
                    // Optional: whether to continue traversing neighbors beyond 2 hops.
                    // Comment out the line below if you only need to collect 2+ hop neighbors without expanding further
                    bfs_queue.push({neighbor_id, next_hop});
                }
            }
        }
    }
    
    return result;
}


// ============ Training Dataset Structures ============
struct LandmarkNeighbors {
    int landmark_id;                            // landmark point id
    std::vector<float> landmark_vector;         // landmark point vector
    std::vector<std::vector<float>> hop1_vectors;   // hop1 neighbor vectors
    std::vector<std::vector<float>> hop2_vectors;   // hop2 neighbor vectors
    std::vector<std::vector<float>> hop3_vectors;   // hop3 neighbor vectors
};

struct TrainingDataset {
    int dim;                            // vector dimension
    std::vector<LandmarkNeighbors> landmarks;  // all landmark data
};

// ============ Binary File Operations ============
/**
 * Save training dataset in binary format
 * Format description:
 * [Header]
 * - dim (uint32_t): vector dimension
 * - num_landmarks (uint32_t): number of landmarks
 * 
 * [Landmarks section]
 * For each landmark:
 *   - landmark_id (uint32_t)
 *   - landmark_vector: dim floats
 *   - hop1_count (uint32_t)
 *   - hop1_vectors: hop1_count * dim floats
 *   - hop2_count (uint32_t)
 *   - hop2_vectors: hop2_count * dim floats
 *   - hop3_count (uint32_t)
 *   - hop3_vectors: hop3_count * dim floats
 */
void saveTrainingDatasetBinary(const std::string& filename, const TrainingDataset& dataset) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Write header
    uint32_t dim = dataset.dim;
    uint32_t num_landmarks = dataset.landmarks.size();
    
    file.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&num_landmarks), sizeof(uint32_t));
    
    // Write landmarks section
    for (const auto& landmark : dataset.landmarks) {
        uint32_t landmark_id = landmark.landmark_id;
        file.write(reinterpret_cast<const char*>(&landmark_id), sizeof(uint32_t));
        
        // Write landmark vector
        file.write(reinterpret_cast<const char*>(landmark.landmark_vector.data()), dim * sizeof(float));
        
        // Write hop1
        uint32_t hop1_count = landmark.hop1_vectors.size();
        file.write(reinterpret_cast<const char*>(&hop1_count), sizeof(uint32_t));
        for (const auto& vec : landmark.hop1_vectors) {
            file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
        }
        
        // Write hop2
        uint32_t hop2_count = landmark.hop2_vectors.size();
        file.write(reinterpret_cast<const char*>(&hop2_count), sizeof(uint32_t));
        for (const auto& vec : landmark.hop2_vectors) {
            file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
        }
        
        // Write hop3
        uint32_t hop3_count = landmark.hop3_vectors.size();
        file.write(reinterpret_cast<const char*>(&hop3_count), sizeof(uint32_t));
        for (const auto& vec : landmark.hop3_vectors) {
            file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
        }
    }
    
    file.close();
    std::cout << "✓ Binary file saved: " << filename << " (size: ";
    std::ifstream check(filename, std::ios::binary | std::ios::ate);
    std::cout << check.tellg() / (1024.0 * 1024.0) << " MB)" << std::endl;
}

/**
 * Load training dataset from binary file
 */
TrainingDataset loadTrainingDatasetBinary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    TrainingDataset dataset;
    
    // Read header
    uint32_t num_landmarks;
    file.read(reinterpret_cast<char*>(&dataset.dim), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&num_landmarks), sizeof(uint32_t));
    
    // Read landmarks section
    dataset.landmarks.resize(num_landmarks);
    for (uint32_t i = 0; i < num_landmarks; i++) {
        LandmarkNeighbors& landmark = dataset.landmarks[i];
        uint32_t landmark_id;
        file.read(reinterpret_cast<char*>(&landmark_id), sizeof(uint32_t));
        landmark.landmark_id = landmark_id;
        
        // Read landmark vector
        landmark.landmark_vector.resize(dataset.dim);
        file.read(reinterpret_cast<char*>(landmark.landmark_vector.data()), dataset.dim * sizeof(float));
        
        // Read hop1 vectors
        uint32_t hop1_count;
        file.read(reinterpret_cast<char*>(&hop1_count), sizeof(uint32_t));
        landmark.hop1_vectors.resize(hop1_count, std::vector<float>(dataset.dim));
        for (uint32_t j = 0; j < hop1_count; j++) {
            file.read(reinterpret_cast<char*>(landmark.hop1_vectors[j].data()), dataset.dim * sizeof(float));
        }
        
        // Read hop2 vectors
        uint32_t hop2_count;
        file.read(reinterpret_cast<char*>(&hop2_count), sizeof(uint32_t));
        landmark.hop2_vectors.resize(hop2_count, std::vector<float>(dataset.dim));
        for (uint32_t j = 0; j < hop2_count; j++) {
            file.read(reinterpret_cast<char*>(landmark.hop2_vectors[j].data()), dataset.dim * sizeof(float));
        }
        
        // Read hop3 vectors
        uint32_t hop3_count;
        file.read(reinterpret_cast<char*>(&hop3_count), sizeof(uint32_t));
        landmark.hop3_vectors.resize(hop3_count, std::vector<float>(dataset.dim));
        for (uint32_t j = 0; j < hop3_count; j++) {
            file.read(reinterpret_cast<char*>(landmark.hop3_vectors[j].data()), dataset.dim * sizeof(float));
        }
    }
    
    file.close();
    return dataset;
}

/**
 * Load landmark IDs from file
 * File format: one ID per line
 */
std::vector<int> loadLandmarkIdsFromFile(const std::string& filename) {
    std::vector<int> landmark_ids;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open landmark file: " + filename);
    }
    
    int id;
    while (file >> id) {
        landmark_ids.push_back(id);
    }
    
    file.close();
    return landmark_ids;
}

/**
 * Print training dataset statistics
 */
void printDatasetStatistics(const TrainingDataset& dataset) {
    std::cout << "\n========== Dataset Statistics ===========" << std::endl;
    std::cout << "Vector dimension: " << dataset.dim << std::endl;
    std::cout << "Number of landmarks: " << dataset.landmarks.size() << std::endl;
    
    int total_hop1 = 0, total_hop2 = 0, total_hop3 = 0;
    int max_hop1 = 0, max_hop2 = 0, max_hop3 = 0;
    
    for (const auto& landmark : dataset.landmarks) {
        total_hop1 += landmark.hop1_vectors.size();
        total_hop2 += landmark.hop2_vectors.size();
        total_hop3 += landmark.hop3_vectors.size();
        max_hop1 = std::max(max_hop1, (int)landmark.hop1_vectors.size());
        max_hop2 = std::max(max_hop2, (int)landmark.hop2_vectors.size());
        max_hop3 = std::max(max_hop3, (int)landmark.hop3_vectors.size());
    }
    
    std::cout << "\nHop Statistics:" << std::endl;
    std::cout << "  Hop1 total: " << total_hop1 << " (avg: " << (double)total_hop1 / dataset.landmarks.size() 
              << ", max: " << max_hop1 << ")" << std::endl;
    std::cout << "  Hop2 total: " << total_hop2 << " (avg: " << (double)total_hop2 / dataset.landmarks.size() 
              << ", max: " << max_hop2 << ")" << std::endl;
    std::cout << "  Hop3 total: " << total_hop3 << " (avg: " << (double)total_hop3 / dataset.landmarks.size() 
              << ", max: " << max_hop3 << ")" << std::endl;
}

// Simple example: demonstrate PyTorch distance computation
int main() {
    std::cout << "HNSW Training Dataset Generator" << std::endl;
    std::cout << "========================================" << std::endl;

int dim = 960;  // Match your SiameseNetwork input dimension
    int max_elements = 1000;
    int M = 16;
    int ef_construction = 200;

    // Create L2 space (default)
    hnswlib::L2Space space(dim);

    // Create HNSW index
    hnswlib::HierarchicalNSW<float> hnsw_index(&space, max_elements, M, ef_construction);

    // Generate random test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    std::cout << "\n[Step 1] Adding elements to index..." << std::endl;
    int num_elements = 1000;

    for (int i = 0; i < num_elements; i++) {
        std::vector<float> data(dim);
        for (int j = 0; j < dim; j++) {
            data[j] = dis(gen);
        }
        // label starts from 0
        hnsw_index.addPoint(data.data(), i);
        
        if ((i + 1) % 200 == 0) {
            std::cout << "  Added " << (i + 1) << " elements" << std::endl;
        }
    }

    std::cout << "✓ Added " << num_elements << " elements to index" << std::endl;

    // ======== Step 2: Generate training dataset ========
    std::cout << "\n[Step 2] Generating training dataset..." << std::endl;
    
    TrainingDataset dataset;
    dataset.dim = dim;
    
    // Read landmark points from file
    std::string landmark_file = "landmark_ids.txt";
    std::vector<int> landmark_ids;
    
    try {
        landmark_ids = loadLandmarkIdsFromFile(landmark_file);
        std::cout << "✓ Read " << landmark_ids.size() << " landmark points from file " << landmark_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    // Validate landmark IDs
    for (int id : landmark_ids) {
        if (id < 0 || id >= num_elements) {
            std::cerr << "Error: landmark ID " << id << " out of range [0, " << num_elements - 1 << "]" << std::endl;
            return 1;
        }
    }
    
    int num_landmarks = landmark_ids.size();
    std::cout << "  Landmark IDs: ";
    for (int i = 0; i < std::min(10, (int)landmark_ids.size()); i++) {
        std::cout << landmark_ids[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    // Call findNeighborsByHopRange for each landmark
    int max_hop_neighbors = num_elements * 0.2;  // max neighbors per hop
    
    std::cout << "\n[Step 2b] Finding neighbors for each landmark (max " << max_hop_neighbors << " per hop)..." << std::endl;
    
    for (size_t i = 0; i < landmark_ids.size(); i++) {
        int landmark_id = landmark_ids[i];
        
        // Call findNeighborsByHopRange to find neighbors
        HopRangeNeighbors hop_neighbors = findNeighborsByHopRange(&hnsw_index, landmark_id);
        
        LandmarkNeighbors landmark_data;
        landmark_data.landmark_id = landmark_id;
        
        // Limit number of neighbors per hop
        if (hop_neighbors.hop1.size() > max_hop_neighbors) {
            hop_neighbors.hop1.resize(max_hop_neighbors);
        }
        if (hop_neighbors.hop2.size() > max_hop_neighbors) {
            hop_neighbors.hop2.resize(max_hop_neighbors);
        }
        if (hop_neighbors.hop3.size() > max_hop_neighbors) {
            hop_neighbors.hop3.resize(max_hop_neighbors);
        }
        
        // Store landmark vector
        const float* landmark_data_ptr = reinterpret_cast<const float*>(hnsw_index.getDataByInternalId(landmark_id));
        landmark_data.landmark_vector = std::vector<float>(landmark_data_ptr, landmark_data_ptr + dim);
        
        // Convert ID to vector: hop1
        for (int neighbor_id : hop_neighbors.hop1) {
            const float* neighbor_data_ptr = reinterpret_cast<const float*>(hnsw_index.getDataByInternalId(neighbor_id));
            std::vector<float> neighbor_vec(neighbor_data_ptr, neighbor_data_ptr + dim);
            landmark_data.hop1_vectors.push_back(neighbor_vec);
        }
        
        // Convert ID to vector: hop2
        for (int neighbor_id : hop_neighbors.hop2) {
            const float* neighbor_data_ptr = reinterpret_cast<const float*>(hnsw_index.getDataByInternalId(neighbor_id));
            std::vector<float> neighbor_vec(neighbor_data_ptr, neighbor_data_ptr + dim);
            landmark_data.hop2_vectors.push_back(neighbor_vec);
        }
        
        // Convert ID to vector: hop3
        for (int neighbor_id : hop_neighbors.hop3) {
            const float* neighbor_data_ptr = reinterpret_cast<const float*>(hnsw_index.getDataByInternalId(neighbor_id));
            std::vector<float> neighbor_vec(neighbor_data_ptr, neighbor_data_ptr + dim);
            landmark_data.hop3_vectors.push_back(neighbor_vec);
        }
        
        dataset.landmarks.push_back(landmark_data);
        
        if ((i + 1) % 20 == 0) {
            std::cout << "  Processed " << (i + 1) << "/" << landmark_ids.size() << " landmarks" << std::endl;
        }
    }
    
    std::cout << "✓ Found neighbors for all landmarks" << std::endl;
    
    // ======== Step 3: Save training dataset ========
    std::cout << "\n[Step 3] Saving training dataset..." << std::endl;
    
    std::string binary_file = "training_dataset.bin";
    saveTrainingDatasetBinary(binary_file, dataset);
    
    // Print statistics
    printDatasetStatistics(dataset);
    
    // ======== Step 4: Verify saved data ========
    std::cout << "\n[Step 4] Verifying data integrity..." << std::endl;
    
    try {
        TrainingDataset loaded_dataset = loadTrainingDatasetBinary(binary_file);
        
        if (loaded_dataset.dim == dataset.dim &&
            loaded_dataset.landmarks.size() == dataset.landmarks.size()) {
            
            std::cout << "✓ Data verification passed! Successfully loaded " << binary_file << std::endl;
            std::cout << "  Loaded landmark count: " << loaded_dataset.landmarks.size() << std::endl;
        } else {
            std::cerr << "✗ Data verification failed: data mismatch" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "✗ Load failed: " << e.what() << std::endl;
        return 1;
    }
    
    // std::cout << "\n✓ Complete workflow finished!" << std::endl;
    // std::cout << "\nProcess summary:" << std::endl;
    // std::cout << "  1. Build HNSW index (" << num_elements << " vectors)" << std::endl;
    // std::cout << "  2. Randomly select " << num_landmarks << " landmark points" << std::endl;
    // std::cout << "  3. Find hop range neighbors for each landmark" << std::endl;
    // std::cout << "  4. Save each landmark and its neighbor vectors to binary file: " << binary_file << std::endl;
    // std::cout << "\nStorage format: only contains landmark-related vectors (no full vector storage)" << std::endl;
    // std::cout << "\nNext step: Run Python script to process data to numpy format" << std::endl;
    // std::cout << "  python convert_dataset_to_numpy.py " << binary_file << std::endl;

    // ======== Step 5: Train model ========
    std::string train_command = "python train_model_from_data.py " + "training_dataset.bin" + " siamese_model.pt 20";

    // ======== Step 6: Perform neighbor search ========
    return 0;
}
