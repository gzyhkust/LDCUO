#pragma once

// Include standard libraries and custom headers first
#include "visited_list_pool.h"
#include "hnswlib.h"
#include "basis.h"
#include "embeddings_manager.h"

// Standard library headers (C++ version)
#include <atomic>
#include <random>
#include <cstdlib>
#include <cassert>
#include <unordered_set>
#include <list>
#include <memory>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>

// Conditionally include PyTorch (must be after all standard libraries)
#ifdef USE_PYTORCH
#include "pytorch_distance.h"
#endif

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    float **embeddingVectors = nullptr;  // vectors after embedding
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

#ifdef USE_PYTORCH
    // PyTorch距离计算器
    std::unique_ptr<PyTorchDistanceCalculator> pytorch_calculator_{nullptr};
    bool use_pytorch_distance_{false};
#else
    bool use_pytorch_distance_{false};
#endif

    // Embeddings管理器：用于存储预计算的向量embeddings
    std::unique_ptr<EmbeddingsManager> embeddings_manager_{nullptr};
    bool use_precomputed_embeddings_{false};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;


        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        embeddingVectors = (float **) malloc(max_elements_ * sizeof(float *));
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");
        if (embeddingVectors == nullptr)
            throw std::runtime_error("Not enough memory: failed to allocate embeddingVectors");
        // 初始化所有 embedding 指针为 nullptr
        for (size_t i = 0; i < max_elements_; i++) {
            embeddingVectors[i] = nullptr;
        }

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;


    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        
        // 释放 embeddingVectors 内存
        if (embeddingVectors != nullptr) {
            for (size_t i = 0; i < max_elements_; i++) {
                if (embeddingVectors[i] != nullptr) {
                    free(embeddingVectors[i]);
                    embeddingVectors[i] = nullptr;
                }
            }
            free(embeddingVectors);
            embeddingVectors = nullptr;
        }
        
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        //std::cout << "data_level0_memory_:" << (char)*data_level0_memory_ << "  internal_id:" << internal_id << " size_data_per_element_:" << size_data_per_element_ << " offsetData_ :" << offsetData_ << std::endl;
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    /**
     * 计算embeddingVectors中两个向量之间的L2距离
     * @param id1 第一个向量的ID（embeddingVectors数组索引）
     * @param id2 第二个向量的ID（embeddingVectors数组索引）
     * @param embedding_dim embedding维度（默认32）
     * @return 两个embedding向量之间的L2距离
     */
    inline dist_t computeEmbeddingDistance(tableint id1, tableint id2, uint32_t embedding_dim = 32) const {
        if (!embeddingVectors) {
            std::cerr << "❌ [ERROR] embeddingVectors 未分配内存！需要调用 embedExternalVectors() 或 embedAllVectors()" << std::endl;
            std::cerr << "   提示：请确保已正确初始化 PyTorch 模型和 embedding 向量" << std::endl;
            return std::numeric_limits<dist_t>::max();
        }
        
        if (id1 < 0 || id1 >= (tableint)cur_element_count || id2 < 0 || id2 >= (tableint)cur_element_count) {
            std::cerr << "❌ [ERROR] 无效的向量ID: id1=" << id1 << ", id2=" << id2 
                      << ", 有效范围: [0, " << (cur_element_count - 1) << "]" << std::endl;
            return std::numeric_limits<dist_t>::max();
        }
        
        if (!embeddingVectors[id1]) {
            std::cerr << "❌ [ERROR] embeddingVectors[" << id1 << "] 为 NULL，该向量未进行 embedding" << std::endl;
            std::cerr << "   提示：请先调用 embedExternalVectors() 为所有向量生成 embeddings" << std::endl;
            return std::numeric_limits<dist_t>::max();
        }
        
        if (!embeddingVectors[id2]) {
            std::cerr << "❌ [ERROR] embeddingVectors[" << id2 << "] 为 NULL，该向量未进行 embedding" << std::endl;
            std::cerr << "   提示：请先调用 embedExternalVectors() 为所有向量生成 embeddings" << std::endl;
            return std::numeric_limits<dist_t>::max();
        }

        float* emb1 = embeddingVectors[id1];
        float* emb2 = embeddingVectors[id2];
        
        float dist = 0;
        for (uint32_t i = 0; i < embedding_dim; i++) {
            float diff = emb1[i] - emb2[i];
            dist += diff * diff;
        }
        
        return static_cast<dist_t>(dist);  // 返回平方距离（L2距离）
    }

    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnnLDCUO(Query &query, BaseFilterFunctor* isIdAllowed = nullptr) {
        std::priority_queue<std::pair<dist_t, labeltype >> result;

        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;


        for(int level = maxlevel_; level > 0; level--) {
            // 使用embeddingVectors计算距离
            dist_t curdist = computeEmbeddingDistance(query.query_Id, currObj);


            bool changed = true;
            while(changed){
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        std::cout << cand << " ---111 ";
                        //throw std::runtime_error("cand error");

                    // 使用embeddingVectors计算距离
                    dist_t d = computeEmbeddingDistance(query.query_Id, cand);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }                
            }
        }

        //std::cout << "entry point: " << currObj << " current dis: " << getDisByLevel(query.query_Id, currObj, 0) << "   " << std::max(ef_, (size_t)query.k) << std::endl;
        
        // Search at the base layer.
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            //std::cout << " bare_bone_search !num_deleted_ && !isIdAllowed " << std::endl;
            top_candidates = searchBaseLayerLDCUO<true>(
                    currObj, query.query_Id, (size_t)(query.k*1), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerLDCUO<false>(
                    currObj, query.query_Id, (size_t)(query.k*1), isIdAllowed);
        }

        while (top_candidates.size() > query.k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    /**
     * 计算两个向量之间的距离（通过指针）
     * 支持三种模式：
     * 1. 预计算embeddings模式：不使用此接口（应该使用computeDistanceById）
     * 2. PyTorch在线模式：使用PyTorch模型实时embedding
     * 3. 标准L2模式：使用传统距离函数
     */
    inline dist_t computeDistance(const void *data_point1, const void *data_point2) const {
        // 如果启用了预计算embeddings，则不应该使用此接口
        if (use_precomputed_embeddings_) {
            std::cerr << "⚠️ 警告：处于预计算embeddings模式，应该使用computeDistanceById而不是computeDistance" << std::endl;
            return fstdistfunc_(data_point1, data_point2, dist_func_param_);
        }

#ifdef USE_PYTORCH
        if (use_pytorch_distance_ && pytorch_calculator_ && pytorch_calculator_->isModelLoaded()) {
            // 使用PyTorch模型计算距离
            try {
                const float* vec1 = static_cast<const float*>(data_point1);
                const float* vec2 = static_cast<const float*>(data_point2);
                
                float pytorch_dist = pytorch_calculator_->computeDistance(vec1, vec2, 960);

                return static_cast<dist_t>(pytorch_dist);
            } catch (const std::exception& e) {
                std::cerr << "PyTorch distance calculation failed, falling back to standard metric: " << e.what() << std::endl;
                return fstdistfunc_(data_point1, data_point2, dist_func_param_);
            }
        }
#endif
        // 使用传统的距离函数
        return fstdistfunc_(data_point1, data_point2, dist_func_param_);
    }

    /**
     * 计算两个向量之间的距离（通过internal ID）
     * 在启用预计算embeddings模式下使用，直接从embeddings_manager中查询embedding
     * 性能最优，避免重复embedding计算
     * 
     * 特殊用法：当id1 = cur_element_count时，表示查询向量的embedding应该存储在虚拟ID cur_element_count处
     * 
     * @param id1 第一个向量的internal ID（或cur_element_count表示查询向量）
     * @param id2 第二个向量的internal ID
     * @return 两个向量之间的距离
     */
    inline dist_t computeDistanceById(tableint id1, tableint id2) const {
        if (!use_precomputed_embeddings_ || !embeddings_manager_) {
            // 回退到原始向量的距离计算
            const void* data1 = getDataByInternalId(id1);
            const void* data2 = getDataByInternalId(id2);
            return computeDistance(data1, data2);
        }

        try {
            const float* emb1 = embeddings_manager_->getEmbedding(id1);
            const float* emb2 = embeddings_manager_->getEmbedding(id2);

            if (!emb1 || !emb2) {
                // 如果embedding不存在，回退到原始向量计算
                const void* data1 = id1 < cur_element_count ? getDataByInternalId(id1) : nullptr;
                const void* data2 = getDataByInternalId(id2);
                if (data1 && data2) {
                    return computeDistance(data1, data2);
                }
                return std::numeric_limits<dist_t>::max();
            }

            // 直接计算embedding之间的L2距离
            uint32_t embedding_dim = embeddings_manager_->getEmbeddingDim();
            float dist = 0;
            for (uint32_t i = 0; i < embedding_dim; i++) {
                float diff = emb1[i] - emb2[i];
                dist += diff * diff;
            }
            return static_cast<dist_t>(dist);

        } catch (const std::exception& e) {
            std::cerr << "Error in computeDistanceById: " << e.what() << std::endl;
            const void* data1 = id1 < cur_element_count ? getDataByInternalId(id1) : nullptr;
            const void* data2 = getDataByInternalId(id2);
            if (data1 && data2) {
                return computeDistance(data1, data2);
            }
            return std::numeric_limits<dist_t>::max();
        }
    }

    /**
     * 初始化PyTorch距离计算器
     * @param model_path TorchScript模型文件路径
     * @param use_cuda 是否使用CUDA加速（如果可用）
     */
#ifdef USE_PYTORCH
    void initPyTorchDistance(const std::string& model_path, bool use_cuda = false) {
        try {
            // 计算实际维度：data_size_是字节数，需要除以sizeof(float)得到维度
            int input_dim = data_size_ / sizeof(float);
            pytorch_calculator_ = std::make_unique<PyTorchDistanceCalculator>(
                model_path, 
                input_dim,  // 维度数，不是字节数
                32,          // embed_dim, 可以根据需要修改
                use_cuda
            );
            use_pytorch_distance_ = true;
            std::cout << "PyTorch distance calculator initialized successfully" << std::endl;
            std::cout << "  Input dimension: " << input_dim << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize PyTorch distance calculator: " << e.what() << std::endl;
            use_pytorch_distance_ = false;
        }
    }

    /**
     * 禁用PyTorch距离计算，回退到传统方法
     */
    void disablePyTorchDistance() {
        use_pytorch_distance_ = false;
        std::cout << "PyTorch distance calculation disabled, using standard metric" << std::endl;
    }
#else
    // Stub methods for non-PyTorch builds
    void initPyTorchDistance(const std::string& model_path, bool use_cuda = false) {
        std::cout << "PyTorch support not compiled in" << std::endl;
    }

    void disablePyTorchDistance() {
        std::cout << "PyTorch support not compiled in" << std::endl;
    }
#endif

    /**
     * 加载预计算的embeddings
     */
    bool loadPrecomputedEmbeddings(const std::string& embeddings_file) {
        if (!embeddings_manager_) {
            embeddings_manager_ = std::make_unique<EmbeddingsManager>();
        }
        
        if (embeddings_manager_->loadFromFile(embeddings_file)) {
            use_precomputed_embeddings_ = true;
            std::cout << "✓ 已启用预计算embeddings，Embedding维度: " << embeddings_manager_->getEmbeddingDim() << std::endl;
            return true;
        }
        return false;
    }

    /**
     * 禁用预计算embeddings
     */
    void disablePrecomputedEmbeddings() {
        use_precomputed_embeddings_ = false;
        if (embeddings_manager_) {
            embeddings_manager_->clear();
        }
        std::cout << "✓ 已禁用预计算embeddings，使用原始向量计算距离" << std::endl;
    }

    /**
     * 使用预计算embeddings计算距离 (通过向量ID)
     */
    inline dist_t computeDistanceWithEmbeddings(uint32_t id1, uint32_t id2) const {
        if (!use_precomputed_embeddings_ || !embeddings_manager_) {
            std::cerr << "⚠️ 预计算embeddings未启用，回退到原始向量计算" << std::endl;
            return 0;  // 应该不会到这里
        }

        const float* emb1 = embeddings_manager_->getEmbedding(id1);
        const float* emb2 = embeddings_manager_->getEmbedding(id2);

        if (!emb1 || !emb2) {
            std::cerr << "❌ 无法获取embedding，ID: " << id1 << ", " << id2 << std::endl;
            return std::numeric_limits<dist_t>::max();
        }

        // 计算L2距离
        uint32_t embedding_dim = embeddings_manager_->getEmbeddingDim();
        float dist = 0;
        for (uint32_t i = 0; i < embedding_dim; i++) {
            float diff = emb1[i] - emb2[i];
            dist += diff * diff;
        }
        return static_cast<dist_t>(dist);  // 返回平方距离
    }

    /**
     * 批量embedding所有向量
     * 在模型训练完成后调用此函数，将所有base向量和query向量转换为embedding形式
     * @return 成功返回true，失败返回false
     */
#ifdef USE_PYTORCH
    bool embedAllVectors() {
        if (!use_pytorch_distance_ || !pytorch_calculator_ || !pytorch_calculator_->isModelLoaded()) {
            std::cerr << "❌ PyTorch模型未初始化，请先调用 initPyTorchDistance()" << std::endl;
            return false;
        }

        if (!embeddings_manager_) {
            try {
                // 创建embeddings管理器，指定embedding维度为32（根据模型配置）
                embeddings_manager_ = std::make_unique<EmbeddingsManager>(
                    max_elements_,
                    pytorch_calculator_->getEmbedDim()
                );
            } catch (const std::exception& e) {
                std::cerr << "❌ 无法创建EmbeddingsManager: " << e.what() << std::endl;
                return false;
            }
        }

        std::cout << "\n[embedAllVectors] 开始批量embedding向量..." << std::endl;
        std::cout << "  总向量数: " << cur_element_count << std::endl;
        std::cout << "  向量维度: " << data_size_ / sizeof(float) << std::endl;
        std::cout << "  embedding维度: " << embeddings_manager_->getEmbeddingDim() << std::endl;

        try {
            // 遍历所有向量进行embedding
            for (tableint i = 0; i < cur_element_count; i++) {
                // 获取原始向量数据
                const void* vec_data = getDataByInternalId(i);
                const float* vec_float = static_cast<const float*>(vec_data);
                int vec_dim = data_size_ / sizeof(float);

                // 创建两个向量的tensors（SiameseNetwork需要两个输入）
                torch::Tensor tensor1 = torch::from_blob(
                    (void*)vec_float, {vec_dim}, torch::kFloat32
                ).clone().unsqueeze(0).to(pytorch_calculator_->getDevice());
                
                torch::Tensor tensor2 = torch::from_blob(
                    (void*)vec_float, {vec_dim}, torch::kFloat32
                ).clone().unsqueeze(0).to(pytorch_calculator_->getDevice());

                // 禁用梯度计算
                torch::NoGradGuard no_grad;

                // 通过模型获取embedding
                std::vector<torch::IValue> inputs{tensor1, tensor2};
                auto outputs = pytorch_calculator_->forward(inputs);
                torch::Tensor embed1 = outputs.toTuple()->elements()[0].toTensor();
                torch::Tensor embed2 = outputs.toTuple()->elements()[1].toTensor();

                // 使用第一个embedding（两个相同）
                torch::Tensor embedding = embed1.squeeze(0).cpu().detach();
                
                // 获取embedding数据指针
                float* embed_ptr = embedding.data_ptr<float>();

                // 存储embedding到管理器
                embeddings_manager_->addEmbedding(i, embed_ptr);

                // 进度提示
                if ((i + 1) % 100 == 0 || i == cur_element_count - 1) {
                    std::cout << "  进度: " << (i + 1) << "/" << cur_element_count 
                              << " (" << std::fixed << std::setprecision(1) 
                              << (100.0 * (i + 1) / cur_element_count) << "%)" << std::endl;
                }
            }

            // 启用预计算embeddings模式
            use_precomputed_embeddings_ = true;
            std::cout << "\n✓ 所有向量embedding完成，已启用预计算embeddings模式" << std::endl;
            std::cout << "  后续距离计算将使用embedding向量而不是原始向量" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "❌ embedding过程中出错: " << e.what() << std::endl;
            return false;
        }
    }
#else
    bool embedAllVectors() {
        std::cerr << "❌ PyTorch support not compiled in" << std::endl;
        return false;
    }
#endif

    /**
     * 将外部的base和query向量进行embedding并存储到embeddingVectors中
     * 支持GPU加速和批量并行处理
     * @param base_vectors: base向量数组 [num_base][dim]
     * @param query_vectors: query向量数组 [num_query][dim]
     * @param num_base: base向量数量
     * @param num_query: query向量数量
     * @param input_dim: 向量维度
     * @param embedding_dim: embedding维度（默认32）
     * @param batch_size: 批处理大小（默认256，GPU加速时建议更大）
     * @return 成功返回true，失败返回false
     */
#ifdef USE_PYTORCH
    bool embedExternalVectors(
        float** base_vectors,
        float** query_vectors,
        size_t num_base,
        size_t num_query,
        size_t input_dim,
        uint32_t embedding_dim = 32,
        size_t batch_size = 256) {
        
        if (!use_pytorch_distance_ || !pytorch_calculator_ || !pytorch_calculator_->isModelLoaded()) {
            std::cerr << "❌ PyTorch模型未初始化，请先调用 initPyTorchDistance()" << std::endl;
            return false;
        }

        if (!embeddingVectors) {
            std::cerr << "❌ embeddingVectors未分配内存" << std::endl;
            return false;
        }

        torch::Device device = pytorch_calculator_->getDevice();
        bool use_cuda = (device.type() == torch::kCUDA);

        std::cout << "\n[embedExternalVectors] 开始embedding外部向量 (GPU加速批量处理)..." << std::endl;
        std::cout << "  Base向量数: " << num_base << std::endl;
        std::cout << "  Query向量数: " << num_query << std::endl;
        std::cout << "  输入维度: " << input_dim << std::endl;
        std::cout << "  输出embedding维度: " << embedding_dim << std::endl;
        std::cout << "  批处理大小: " << batch_size << std::endl;
        std::cout << "  使用设备: " << (use_cuda ? "CUDA GPU" : "CPU") << std::endl;

        size_t total_vectors = num_base + num_query;

        try {
            // 禁用梯度计算
            torch::NoGradGuard no_grad;

            // ========== 批量 Embedding base向量 ==========
            std::cout << "\n[步骤1] 批量Embedding base向量..." << std::endl;
            
            for (size_t batch_start = 0; batch_start < num_base; batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, num_base);
                size_t current_batch_size = batch_end - batch_start;
                
                // 创建批量tensor: [batch_size, input_dim]
                torch::Tensor batch_tensor = torch::zeros({(long)current_batch_size, (long)input_dim}, torch::kFloat32);
                
                // 填充批量数据
                for (size_t i = 0; i < current_batch_size; i++) {
                    torch::Tensor vec_tensor = torch::from_blob(
                        base_vectors[batch_start + i], {(long)input_dim}, torch::kFloat32
                    ).clone();
                    batch_tensor[i] = vec_tensor;
                }
                
                // 移动到GPU（如果可用）
                batch_tensor = batch_tensor.to(device);
                
                // 批量前向传播（SiameseNetwork需要两个输入，这里两个相同）
                std::vector<torch::IValue> inputs{batch_tensor, batch_tensor};
                auto outputs = pytorch_calculator_->forward(inputs);
                
                // 获取embedding结果: [batch_size, embedding_dim]
                torch::Tensor embeddings = outputs.toTuple()->elements()[0].toTensor();
                embeddings = embeddings.cpu().detach();  // 移回CPU
                
                // 将结果复制到embeddingVectors
                float* embeddings_ptr = embeddings.data_ptr<float>();
                for (size_t i = 0; i < current_batch_size; i++) {
                    size_t idx = batch_start + i;
                    embeddingVectors[idx] = (float*) malloc(embedding_dim * sizeof(float));
                    if (!embeddingVectors[idx]) {
                        std::cerr << "❌ 无法为embedding分配内存: 索引 " << idx << std::endl;
                        return false;
                    }
                    memcpy(embeddingVectors[idx], embeddings_ptr + i * embedding_dim, embedding_dim * sizeof(float));
                }
                
                // 进度提示
                if (batch_end % 10000 < batch_size || batch_end == num_base) {
                    std::cout << "  ✓ 已embedding " << batch_end << "/" << num_base 
                              << " 个base向量 ("
                              << std::fixed << std::setprecision(1) 
                              << (100.0 * batch_end / num_base) << "%)" << std::endl;
                }
            }

            // ========== 批量 Embedding query向量 ==========
            if (num_query > 0) {
                std::cout << "\n[步骤2] 批量Embedding query向量..." << std::endl;
                
                for (size_t batch_start = 0; batch_start < num_query; batch_start += batch_size) {
                    size_t batch_end = std::min(batch_start + batch_size, num_query);
                    size_t current_batch_size = batch_end - batch_start;
                    
                    // 创建批量tensor: [batch_size, input_dim]
                    torch::Tensor batch_tensor = torch::zeros({(long)current_batch_size, (long)input_dim}, torch::kFloat32);
                    
                    // 填充批量数据
                    for (size_t i = 0; i < current_batch_size; i++) {
                        torch::Tensor vec_tensor = torch::from_blob(
                            query_vectors[batch_start + i], {(long)input_dim}, torch::kFloat32
                        ).clone();
                        batch_tensor[i] = vec_tensor;
                    }
                    
                    // 移动到GPU（如果可用）
                    batch_tensor = batch_tensor.to(device);
                    
                    // 批量前向传播
                    std::vector<torch::IValue> inputs{batch_tensor, batch_tensor};
                    auto outputs = pytorch_calculator_->forward(inputs);
                    
                    // 获取embedding结果
                    torch::Tensor embeddings = outputs.toTuple()->elements()[0].toTensor();
                    embeddings = embeddings.cpu().detach();
                    
                    // 将结果复制到embeddingVectors
                    float* embeddings_ptr = embeddings.data_ptr<float>();
                    for (size_t i = 0; i < current_batch_size; i++) {
                        size_t idx = num_base + batch_start + i;
                        embeddingVectors[idx] = (float*) malloc(embedding_dim * sizeof(float));
                        if (!embeddingVectors[idx]) {
                            std::cerr << "❌ 无法为embedding分配内存: 索引 " << idx << std::endl;
                            return false;
                        }
                        memcpy(embeddingVectors[idx], embeddings_ptr + i * embedding_dim, embedding_dim * sizeof(float));
                    }
                    
                    // 进度提示
                    if (batch_end % 1000 < batch_size || batch_end == num_query) {
                        std::cout << "  ✓ 已embedding " << batch_end << "/" << num_query 
                                  << " 个query向量 ("
                                  << std::fixed << std::setprecision(1) 
                                  << (100.0 * batch_end / num_query) << "%)" << std::endl;
                    }
                }
            }

            std::cout << "\n✓ 所有向量embedding完成！" << std::endl;
            std::cout << "  总计: " << total_vectors << " 个向量已embedding" << std::endl;
            std::cout << "  Base向量: [0, " << num_base - 1 << "]" << std::endl;
            if (num_query > 0) {
                std::cout << "  Query向量: [" << num_base << ", " << (num_base + num_query - 1) << "]" << std::endl;
            }
            
            return true;

        } catch (const std::exception& e) {
            std::cerr << "❌ embedding过程中出错: " << e.what() << std::endl;
            return false;
        }
    }
#else
    bool embedExternalVectors(
        float** base_vectors,
        float** query_vectors,
        size_t num_base,
        size_t num_query,
        size_t input_dim,
        uint32_t embedding_dim = 32) {
        std::cerr << "❌ PyTorch support not compiled in" << std::endl;
        return false;
    }
#endif



    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr,
        tableint query_id = (tableint)-1) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            // 如果提供了query_id且embeddingVectors可用，则使用embeddingVectors计算距离
            dist_t dist;
            if (query_id != (tableint)-1 && embeddingVectors && embeddingVectors[query_id]) {
                dist = computeEmbeddingDistance(query_id, ep_id);
            } else {
                // 回退到原始计算方式
                dist = use_precomputed_embeddings_ ? computeDistanceById(cur_element_count, ep_id) : computeDistance(data_point, ep_data);
            }
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    // 如果提供了query_id且embeddingVectors可用，则使用embeddingVectors计算距离
                    dist_t dist;
                    if (query_id != (tableint)-1 && embeddingVectors && embeddingVectors[query_id]) {
                        dist = computeEmbeddingDistance(query_id, candidate_id);
                    } else {
                        // 回退到原始计算方式
                        dist = use_precomputed_embeddings_ ? computeDistanceById(cur_element_count, candidate_id) : computeDistance(data_point, currObj1);
                    }

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

     // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerLDCUO(
        tableint ep_id,
        tableint query,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = computeEmbeddingDistance(ep_id, query);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = getDataByInternalId(candidate_id);
                    // 使用computeDistanceById如果启用了预计算embeddings，否则使用原始computeDistance
                    dist_t dist = computeEmbeddingDistance(candidate_id, query);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }   


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        //std::cout << "The first addPoint Alg " << std::endl;
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        //std::cout << "The second addPoint Alg " << std::endl;
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        //std::cout << "curlevel: " << curlevel<< " maxlevel_:" <<maxlevel_<< std::endl;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        //std::cout << "cur_c : " << cur_c << " " << "label: " << label <<  std::endl;
        //std::cout << getExternalLabeLp(cur_c) << " " << getDataByInternalId(cur_c) <<std::endl;
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            //std::cout << " cur_c: " << cur_c  << " curlevel:" << curlevel << " linkLists_[cur_c].size:" << size_links_per_element_ * curlevel + 1 << std::endl;
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        //std::cout << "currObj: " << currObj << "  " << "enterpoint_copy: " << enterpoint_copy << std::endl;
        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                //std::cout << "data_point : " << data_point << " " << "currObj: " << currObj <<  std::endl;
                // std::cout << "getDataByInternalId() : " << typeid(getDataByInternalId(currObj)).name() << " " << "curdist: " << curdist <<  std::endl;
                // unsigned char* a = (unsigned char*)getDataByInternalId(currObj);
                // for(int z= 0; z<8; z++){
                //     std::cout << (double)*a <<  "  ";
                //     a++;
                // }
                
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        //std::cout << "get_linklist: " << currObj << " " << level << " " << get_linklist(currObj, level) << " " << *get_linklist(currObj, level) <<  " " << getListCount(get_linklist(currObj, level)) <<   std::endl;
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        //std::cout << "datal: " << *datal << std::endl;
                        //for(int i =0; i<size; i++) std::cout<< "i:" << i << " datal[i]" << data[i] << " ";
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, tableint query_Id,  size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        std::cout<< "Start the first phase of searchKnn for query_Id: " << query_Id << std::endl;
        dist_t curdist = computeDistance(query_data, getDataByInternalId(enterpoint_node_));
        std::cout<< " =========================================================== " << std::endl;


        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = computeDistance(query_data, getDataByInternalId(cand));

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }


 
};
}  // namespace hnswlib
