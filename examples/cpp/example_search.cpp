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
#include "../../hnswlib/hnswlib.h"

// Struct: neighbors classified by hop distance
struct HopRangeNeighbors {
    std::vector<int> hop1;      // neighbors within 0-10 hops
    std::vector<int> hop2;     // neighbors within 10-20 hops
    std::vector<int> hop3;      // neighbors beyond 20 hops
};

/**
 * Find neighbors by hop distance in layer 0 graph structure
 * @param alg_hnsw HNSW index pointer
 * @param start_id ID of the starting node
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
                if (next_hop <= 10) {
                    result.hop1.push_back(neighbor_id);
                    bfs_queue.push({neighbor_id, next_hop});
                } else if (next_hop <= 20) {
                    result.hop2.push_back(neighbor_id);
                    bfs_queue.push({neighbor_id, next_hop});
                } else {
                    result.hop3.push_back(neighbor_id);
                    bfs_queue.push({neighbor_id, next_hop});
                }
            }
        }
    }
    
    return result;
}


// ============ Training Dataset Structures ============
struct LandmarkNeighbors {
    int landmark_id;                            // ID of the landmark point
    std::vector<float> landmark_vector;         // vector of the landmark point
    std::vector<std::vector<float>> hop1_vectors;   // vectors of hop1 neighbors
    std::vector<std::vector<float>> hop2_vectors;   // vectors of hop2 neighbors
    std::vector<std::vector<float>> hop3_vectors;   // vectors of hop3 neighbors
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
    
    // Write landmark section
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
    
    // Read landmark section
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



int main(int argc, char** argv) {


    // ------------------------------  STEP 1: Loading the data ------------------------------  
    int dim = -1;               // Dimension of the elements
    int num_base = -1;   // Maximum number of elements, should be known beforehand
    int M = 48;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    int epochs = 200;            // Number of epochs for training

    std::string algo(argv[1]);
    std::string dataset(argv[2]);
    int K = std::stoi(argv[3]); 

    std::string file = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +".data_new";
    //std::string file = "/home/data/zgongae/VectorsIndex/HEDS/" + dataset + "/" + dataset +".data_new";
    std::ifstream loadin(file.c_str(), std::ios::binary);
    while (!loadin) {
        printf("Fail to find data file!\n");
        exit(0);
    }

    unsigned int header[3] = {};
    loadin.read((char*)header, sizeof(header));

    int num_query = -1;
    float **queryVectors = nullptr;
    int **groundtruth = nullptr;

    if(dataset == "openai" || dataset == "msturing" ){
        load_query_groundtruth_(dataset, num_query, queryVectors, groundtruth);
    } else if (dataset == "deep100m" || dataset == "sift100m"){
        load_query_groundtruth100m(dataset, num_query, queryVectors, groundtruth);
    }else{
        load_query_groundtruth(dataset, num_query, queryVectors, groundtruth);
    }

    // ------------------------------  STEP 2: Initial the index ------------------------------ 
    num_base = header[1];
    //num_base = 3000;
    dim = header[2];
    
    // Calculate numLandmarks after num_base is determined
    int numLandmarks = std::min(1024, (int)(num_base * 0.01)); // Number of landmarks to use for training
    std::cout << "numLandmarks: " << numLandmarks << " (" << (numLandmarks * 100.0 / num_base) << "% of base)" << std::endl;

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, num_base + num_query, M, ef_construction);


    // ------------------------------  STEP 3: Build the index ------------------------------ 
    Performance per;
    Timer t;
    
    float** data = new float* [num_base];
    for(int i = 0; i< num_base; ++i){
        data[i] = new float[dim];
        loadin.read((char*)data[i], sizeof(float) * header[2]);          
    }

    std::cout << std::endl;

    t.restart();
    for(int i = 0; i< num_base; ++i){
        if(algo == "hnsw"){
            //std::cout << "Insert: " << i << std::endl;
            alg_hnsw->addPoint(data[i], i, 1);
            if ((i + 1) % 20000 == 0) {
                std::cout << "  Added " << (i + 1) << " elements" << std::endl;
            }   
        }   
    }

    per.setTimeBuildindex(t.elapsed());

    std::string construction_res_path = "/home/data/zgongae/VectorsIndex/LDCUO/results/" + dataset + "_" + algo +"_construction.txt";
    std::ofstream construction_res(construction_res_path);

    std::cout << "Build Index Time : " << per.getTimeBuildindex() << " [s]" << std::endl;
    construction_res << "Build Index Time : " << per.getTimeBuildindex() << " [s]" << std::endl;
    std::cout << "Preprocessing Time: " << per.getTimePreprocessing() << " [s]" << std::endl;
    construction_res << "Preprocessing Time: " << per.getTimePreprocessing() << " [s]" << std::endl;

    t.restart();
    per.setTimeShortcut(t.elapsed());

    // std::cout << "Tree height: " << alg_hnsw->maxlevel_ << std::endl;
    construction_res << "Levels of HNSW: " << alg_hnsw->maxlevel_ << std::endl;
    //std::cout << "Memory cost: " << (alg_hnsw->indexFileSize(num_base))/ (1024.0 * 1024.0) << std::endl;
    //construction_res << "Memory cost: " << (alg_hnsw->indexFileSize(num_base))/ (1024.0 * 1024.0) << std::endl;
    //construction_res << "Memory cost of shortcuts: " << (alg_hnsw->Shortcuts.size_in_bytes()) << std::endl;


    // ------------------------------  STEP 4: Generate training dataset ------------------------------
    TrainingDataset training_dataset;
    training_dataset.dim = dim;

    std::default_random_engine gen;
    gen.seed(std::time(nullptr));
    
    // Select landmark points
    int num_landmarks = numLandmarks;
    std::vector<int> landmark_ids;
    std::vector<int> indices(num_base);
    for (int i = 0; i < num_base; i++) {
        indices[i] = i;
    }
    
    for (int i = num_base - 1; i > num_base - 1 - num_landmarks; i--) {
        int j = gen() % (i + 1);
        std::swap(indices[i], indices[j]);
    }
    
    for (int i = 0; i < num_landmarks; i++) {
        landmark_ids.push_back(indices[num_base - 1 - i]);
    }
    
    std::cout << "✓ Selected " << num_landmarks << " landmark points" << std::endl;
    std::cout << "  Landmark IDs: ";
    for (int i = 0; i < std::min(10, (int)landmark_ids.size()); i++) {
        std::cout << landmark_ids[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    // Call findNeighborsByHopRange for each landmark
    int max_hop_neighbors = 4096;  // maximum neighbors per hop
    
    std::cout << "\n[Step 2b] Finding neighbors for each landmark (max " << max_hop_neighbors << " per hop)..." << std::endl;
    
    for (size_t i = 0; i < landmark_ids.size(); i++) {
        int landmark_id = landmark_ids[i];
        
        // Call findNeighborsByHopRange to find neighbors
        HopRangeNeighbors hop_neighbors = findNeighborsByHopRange(alg_hnsw, landmark_id);
        
        LandmarkNeighbors landmark_data;
        landmark_data.landmark_id = landmark_id;
        
        // Limit the number of neighbors per hop
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
        const float* landmark_data_ptr = reinterpret_cast<const float*>(alg_hnsw->getDataByInternalId(landmark_id));
        landmark_data.landmark_vector = std::vector<float>(landmark_data_ptr, landmark_data_ptr + dim);
        
        // Convert IDs to vectors: hop1
        for (int neighbor_id : hop_neighbors.hop1) {
            const float* neighbor_data_ptr = reinterpret_cast<const float*>(alg_hnsw->getDataByInternalId(neighbor_id));
            std::vector<float> neighbor_vec(neighbor_data_ptr, neighbor_data_ptr + dim);
            landmark_data.hop1_vectors.push_back(neighbor_vec);
        }
        
        // Convert IDs to vectors: hop2
        for (int neighbor_id : hop_neighbors.hop2) {
            const float* neighbor_data_ptr = reinterpret_cast<const float*>(alg_hnsw->getDataByInternalId(neighbor_id));
            std::vector<float> neighbor_vec(neighbor_data_ptr, neighbor_data_ptr + dim);
            landmark_data.hop2_vectors.push_back(neighbor_vec);
        }
        
        // Convert IDs to vectors: hop3
        for (int neighbor_id : hop_neighbors.hop3) {
            const float* neighbor_data_ptr = reinterpret_cast<const float*>(alg_hnsw->getDataByInternalId(neighbor_id));
            std::vector<float> neighbor_vec(neighbor_data_ptr, neighbor_data_ptr + dim);
            landmark_data.hop3_vectors.push_back(neighbor_vec);
        }
        
        training_dataset.landmarks.push_back(landmark_data);
        
        if ((i + 1) % 2000 == 0) {
            std::cout << "  Processed " << (i + 1) << "/" << landmark_ids.size() << " landmarks" << std::endl;
        }
    }
    
    std::cout << "✓ Found neighbors for all landmarks" << std::endl;
    

    std::cout << "\n[Step 3] Training dataset statistics..." << std::endl;
    std::cout << "  Training dataset statistics: " << std::endl;
    std::cout << "    - dim: " << training_dataset.dim << std::endl;
    std::cout << "    - number of landmarks: " << training_dataset.landmarks.size() << std::endl;
    if (!training_dataset.landmarks.empty()) {
        std::cout << "    - hop1 vectors of first landmark: " << training_dataset.landmarks[0].hop1_vectors.size() << std::endl;
        std::cout << "    - hop2 vectors of first landmark: " << training_dataset.landmarks[0].hop2_vectors.size() << std::endl;
        std::cout << "    - hop3 vectors of first landmark: " << training_dataset.landmarks[0].hop3_vectors.size() << std::endl;
    }

    // ------------------------------  STEP 4: Train the model (pipe data, no disk write) ------------------------------
    std::cout << "\n[Step 4] Piping data and training model (no disk write)..." << std::endl;
    
    // Build training command using --stdin parameter
    std::string train_command = std::string("python train_model_from_data.py --stdin siamese_model.pt ") + std::to_string(epochs);
    std::cout << "  Training command: " << train_command << std::endl;
    
    // Record training start time
    t.restart();
    
    // Use popen to open pipe
    FILE* pipe = popen(train_command.c_str(), "w");
    if (!pipe) {
        std::cerr << "❌ Cannot start Python training process!" << std::endl;
        exit(1);
    }
    
    // Write binary data through pipe
    std::cout << "  Piping training data..." << std::endl;
    
    // Write header
    uint32_t write_dim = training_dataset.dim;
    uint32_t write_num_landmarks = training_dataset.landmarks.size();
    fwrite(&write_dim, sizeof(uint32_t), 1, pipe);
    fwrite(&write_num_landmarks, sizeof(uint32_t), 1, pipe);
    
    // Write landmark section
    for (const auto& landmark : training_dataset.landmarks) {
        uint32_t landmark_id = landmark.landmark_id;
        fwrite(&landmark_id, sizeof(uint32_t), 1, pipe);
        
        // Write landmark vector
        fwrite(landmark.landmark_vector.data(), sizeof(float), write_dim, pipe);
        
        // Write hop1
        uint32_t hop1_count = landmark.hop1_vectors.size();
        fwrite(&hop1_count, sizeof(uint32_t), 1, pipe);
        for (const auto& vec : landmark.hop1_vectors) {
            fwrite(vec.data(), sizeof(float), write_dim, pipe);
        }
        
        // Write hop2
        uint32_t hop2_count = landmark.hop2_vectors.size();
        fwrite(&hop2_count, sizeof(uint32_t), 1, pipe);
        for (const auto& vec : landmark.hop2_vectors) {
            fwrite(vec.data(), sizeof(float), write_dim, pipe);
        }
        
        // Write hop3
        uint32_t hop3_count = landmark.hop3_vectors.size();
        fwrite(&hop3_count, sizeof(uint32_t), 1, pipe);
        for (const auto& vec : landmark.hop3_vectors) {
            fwrite(vec.data(), sizeof(float), write_dim, pipe);
        }
    }
    
    // Flush and close pipe, wait for Python process to complete
    fflush(pipe);
    int train_result = pclose(pipe);
    
    if (train_result != 0) {
        std::cerr << "❌ Model training failed! Return code: " << train_result << std::endl;
        exit(1);
    }
    
    double train_time = t.elapsed();
    std::cout << "✓ Model training completed, time: " << train_time << " [s]" << std::endl;
    
    // Write training time to construction_res file
    construction_res << "Model Training Time: " << train_time << " [s]" << std::endl;
    construction_res.close();

    // ------------------------------  STEP 5: Generate embeddings for all vectors ------------------------------
    std::cout << "\n[Step 5] Generating embeddings for all vectors..." << std::endl;

#ifdef USE_PYTORCH
    // Initialize PyTorch distance calculator
    std::string model_path = "siamese_model.pt";
    bool use_cuda = true;  // Set to true for GPU acceleration
    alg_hnsw->initPyTorchDistance(model_path, use_cuda);
    
    // Call embedExternalVectors to embed base and query vectors
    // batch_size: larger batch_size recommended for GPU acceleration (256-1024)
    size_t batch_size = use_cuda ? 512 : 64;  // GPU uses 512, CPU uses 64
    bool embedding_success = alg_hnsw->embedExternalVectors(
        data,              // base vector array
        queryVectors,      // query vector array
        num_base,          // number of base vectors
        num_query,         // number of query vectors
        dim,               // vector dimension
        32,                // embedding dimension
        batch_size         // batch size
    );

    if (!embedding_success) {
        std::cerr << "❌ Embedding failed!" << std::endl;
        exit(1);
    }

    std::cout << "✓ Embedding completed, all vectors stored in embeddingVectors" << std::endl;
#else
    std::cerr << "⚠️ PyTorch support not compiled, skipping embedding step" << std::endl;
#endif

    //return 0;


    // ------------------------------  STEP 6: Load the query and groundtruth ------------------------------
    std::string search_res_path = "/home/data/zgongae/VectorsIndex/LDCUO/results/" + dataset + "_" + algo + "_"+std::to_string(K)+"_search.txt";
    std::ofstream search_res(search_res_path);

    for(int i = 0; i< num_query; i++){
        alg_hnsw->addPoint(queryVectors[i], num_base+i, -1);

        Query query(num_base+i, K);
        std::cout << "query:" << i << std::endl;
        t.restart();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
        result = alg_hnsw->searchKnnLDCUO(query);
        query.setQueryTime(t.elapsed());
        query.setRecall(calculateRecall(i, K, result, groundtruth));

        std:: cout << " time: " << query.getQueryTime() << " Recall: " << query.getRecall()  << std::endl;
        search_res << query.getQueryTime() << " " << query.getRecall()  << std::endl;
        //search_res << query.getQueryTime() << " " << query.getRecall() << " " << query.getNumofDis() << " " << query.getNumofDisBase() << " " << float(query.getNumofScanned()/query.getNumofDis()) << " " << query.getNumofLevelsSkip() << std::endl;
        //search_res << query.getQueryTime() << " " << query.getRecall() << " " << query.getNumofDis() << " " << query.getNumofDisBase() << " " << alg_hnsw->purneNumDis << " " << alg_hnsw->purneBaseNumDis <<" " << query.getNumofLevelsSkip()<<" " << float(query.getNumofScanned()/query.getNumofDis()) << std::endl;
        //std::fill(alg_hnsw->resultsProcessing.begin(), alg_hnsw->resultsProcessing.end(),-1);
    }
    search_res.close();

    clear_2d_array(queryVectors, num_query);
    clear_2d_array(groundtruth, num_query);
    clear_2d_array(data, num_base);
    delete alg_hnsw;
    return 0;
}