#include "embedding.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>

namespace VectorDB {
    
    // HashEmbedding implementation
    HashEmbedding::HashEmbedding(size_t dimension) 
        : dimension_(dimension), rng_(std::random_device{}()) {}
    
    Vector HashEmbedding::generate_embedding(const std::string& text) {
        Vector embedding(dimension_, 0.0f);
        
        // Use multiple hash functions to fill the vector
        for (size_t i = 0; i < dimension_; ++i) {
            std::string modified_text = text + std::to_string(i);
            size_t hash_val = hasher_(modified_text);
            
            // Convert hash to float in range [-1, 1]
            embedding[i] = static_cast<float>(hash_val % 2000 - 1000) / 1000.0f;
        }
        
        return embedding;
    }
    
    Vector HashEmbedding::generate_embedding(const std::vector<float>& features) {
        Vector embedding(dimension_, 0.0f);
        
        for (size_t i = 0; i < dimension_; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < features.size(); ++j) {
                sum += features[j] * static_cast<float>((i * features.size() + j) % 1000);
            }
            embedding[i] = std::tanh(sum / 1000.0f); // Normalize to [-1, 1]
        }
        
        return embedding;
    }
    
    // RandomProjectionEmbedding implementation
    RandomProjectionEmbedding::RandomProjectionEmbedding(size_t input_dim, size_t output_dim)
        : input_dimension_(input_dim), output_dimension_(output_dim), rng_(std::random_device{}()) {
        initialize_projection_matrix();
    }
    
    void RandomProjectionEmbedding::initialize_projection_matrix() {
        projection_matrix_.resize(output_dimension_);
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(input_dimension_)));
        
        for (size_t i = 0; i < output_dimension_; ++i) {
            projection_matrix_[i].resize(input_dimension_);
            for (size_t j = 0; j < input_dimension_; ++j) {
                projection_matrix_[i][j] = dist(rng_);
            }
        }
    }
    
    Vector RandomProjectionEmbedding::generate_embedding(const std::string& text) {
        // Convert text to feature vector (simple character frequency)
        std::vector<float> features(256, 0.0f); // ASCII characters
        for (char c : text) {
            features[static_cast<unsigned char>(c)]++;
        }
        
        // Normalize
        float sum = 0.0f;
        for (float f : features) sum += f;
        if (sum > 0) {
            for (float& f : features) f /= sum;
        }
        
        return generate_embedding(features);
    }
    
    Vector RandomProjectionEmbedding::generate_embedding(const std::vector<float>& features) {
        if (features.size() != input_dimension_) {
            throw DimensionMismatchException("Input features dimension mismatch");
        }
        
        Vector embedding(output_dimension_, 0.0f);
        
        for (size_t i = 0; i < output_dimension_; ++i) {
            for (size_t j = 0; j < input_dimension_; ++j) {
                embedding[i] += features[j] * projection_matrix_[i][j];
            }
        }
        
        return embedding;
    }
    
    // TFIDFEmbedding implementation
    TFIDFEmbedding::TFIDFEmbedding(size_t max_features) 
        : max_features_(max_features), is_fitted_(false) {}
    
    std::vector<std::string> TFIDFEmbedding::tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;
        
        while (iss >> token) {
            // Convert to lowercase and remove punctuation
            std::transform(token.begin(), token.end(), token.begin(), ::tolower);
            token.erase(std::remove_if(token.begin(), token.end(), ::ispunct), token.end());
            
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
        
        return tokens;
    }
    
    void TFIDFEmbedding::fit(const std::vector<std::string>& documents) {
        build_vocabulary(documents);
        calculate_idf(documents);
        is_fitted_ = true;
    }
    
    void TFIDFEmbedding::build_vocabulary(const std::vector<std::string>& documents) {
        std::unordered_map<std::string, size_t> word_counts;
        
        // Count word frequencies across all documents
        for (const auto& doc : documents) {
            auto tokens = tokenize(doc);
            for (const auto& token : tokens) {
                word_counts[token]++;
            }
        }
        
        // Select top max_features_ words
        std::vector<std::pair<std::string, size_t>> word_freq_pairs(word_counts.begin(), word_counts.end());
        std::sort(word_freq_pairs.begin(), word_freq_pairs.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        vocabulary_.clear();
        for (size_t i = 0; i < std::min(max_features_, word_freq_pairs.size()); ++i) {
            vocabulary_[word_freq_pairs[i].first] = i;
        }
    }
    
    void TFIDFEmbedding::calculate_idf(const std::vector<std::string>& documents) {
        std::unordered_map<std::string, size_t> doc_counts;
        
        // Count documents containing each word
        for (const auto& doc : documents) {
            auto tokens = tokenize(doc);
            std::unordered_set<std::string> unique_tokens(tokens.begin(), tokens.end());
            
            for (const auto& token : unique_tokens) {
                if (vocabulary_.find(token) != vocabulary_.end()) {
                    doc_counts[token]++;
                }
            }
        }
        
        // Calculate IDF scores
        size_t total_docs = documents.size();
        for (const auto& pair : vocabulary_) {
            const std::string& word = pair.first;
            size_t doc_freq = doc_counts[word];
            idf_scores_[word] = std::log(static_cast<float>(total_docs) / (1.0f + doc_freq));
        }
    }
    
    Vector TFIDFEmbedding::generate_embedding(const std::string& text) {
        if (!is_fitted_) {
            throw std::runtime_error("TF-IDF model not fitted");
        }
        
        Vector embedding(max_features_, 0.0f);
        auto tokens = tokenize(text);
        
        // Calculate term frequencies
        std::unordered_map<std::string, size_t> term_counts;
        for (const auto& token : tokens) {
            term_counts[token]++;
        }
        
        // Calculate TF-IDF scores
        for (const auto& pair : term_counts) {
            const std::string& word = pair.first;
            size_t count = pair.second;
            
            auto vocab_it = vocabulary_.find(word);
            if (vocab_it != vocabulary_.end()) {
                size_t index = vocab_it->second;
                float tf = static_cast<float>(count) / tokens.size();
                float idf = idf_scores_.at(word);
                embedding[index] = tf * idf;
            }
        }
        
        return embedding;
    }
    
    Vector TFIDFEmbedding::generate_embedding(const std::vector<float>& features) {
        // For TF-IDF, we expect text input, so this is a simple passthrough
        if (features.size() != max_features_) {
            throw DimensionMismatchException("Feature dimension mismatch for TF-IDF");
        }
        return Vector(features.begin(), features.end());
    }
    
    // NGramEmbedding implementation
    NGramEmbedding::NGramEmbedding(size_t n, size_t dimension) 
        : n_(n), dimension_(dimension) {}
    
    std::vector<std::string> NGramEmbedding::generate_ngrams(const std::string& text) const {
        std::vector<std::string> ngrams;
        std::string clean_text = text;
        std::transform(clean_text.begin(), clean_text.end(), clean_text.begin(), ::tolower);
        
        if (clean_text.length() < n_) {
            ngrams.push_back(clean_text);
            return ngrams;
        }
        
        for (size_t i = 0; i <= clean_text.length() - n_; ++i) {
            ngrams.push_back(clean_text.substr(i, n_));
        }
        
        return ngrams;
    }
    
    Vector NGramEmbedding::generate_embedding(const std::string& text) {
        Vector embedding(dimension_, 0.0f);
        auto ngrams = generate_ngrams(text);
        
        for (const auto& ngram : ngrams) {
            size_t hash_val = hasher_(ngram);
            size_t index = hash_val % dimension_;
            embedding[index] += 1.0f;
        }
        
        // Normalize
        float sum = 0.0f;
        for (float val : embedding) sum += val;
        if (sum > 0) {
            for (float& val : embedding) val /= sum;
        }
        
        return embedding;
    }
    
    Vector NGramEmbedding::generate_embedding(const std::vector<float>& features) {
        // Convert features to string representation for n-gram generation
        std::string feature_str;
        for (float f : features) {
            feature_str += std::to_string(static_cast<int>(f * 100)) + " ";
        }
        return generate_embedding(feature_str);
    }
    
    // AverageWordEmbedding implementation
    AverageWordEmbedding::AverageWordEmbedding(size_t dimension) 
        : dimension_(dimension), rng_(std::random_device{}()) {}
    
    std::vector<std::string> AverageWordEmbedding::tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        std::istringstream iss(text);
        std::string token;
        
        while (iss >> token) {
            std::transform(token.begin(), token.end(), token.begin(), ::tolower);
            token.erase(std::remove_if(token.begin(), token.end(), ::ispunct), token.end());
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
        
        return tokens;
    }
    
    Vector AverageWordEmbedding::get_word_embedding(const std::string& word) {
        auto it = word_embeddings_.find(word);
        if (it != word_embeddings_.end()) {
            return it->second;
        }
        
        // Generate random embedding for unknown words
        Vector embedding(dimension_);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < dimension_; ++i) {
            embedding[i] = dist(rng_);
        }
        
        word_embeddings_[word] = embedding;
        return embedding;
    }
    
    void AverageWordEmbedding::add_word_embedding(const std::string& word, const Vector& embedding) {
        if (embedding.size() != dimension_) {
            throw DimensionMismatchException("Word embedding dimension mismatch");
        }
        word_embeddings_[word] = embedding;
    }
    
    void AverageWordEmbedding::load_word_embeddings(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open word embeddings file: " + file_path);
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string word;
            iss >> word;
            
            Vector embedding(dimension_);
            for (size_t i = 0; i < dimension_; ++i) {
                iss >> embedding[i];
            }
            
            word_embeddings_[word] = embedding;
        }
    }
    
    Vector AverageWordEmbedding::generate_embedding(const std::string& text) {
        auto tokens = tokenize(text);
        if (tokens.empty()) {
            return Vector(dimension_, 0.0f);
        }
        
        Vector embedding(dimension_, 0.0f);
        for (const auto& token : tokens) {
            auto word_emb = get_word_embedding(token);
            for (size_t i = 0; i < dimension_; ++i) {
                embedding[i] += word_emb[i];
            }
        }
        
        // Average the embeddings
        for (size_t i = 0; i < dimension_; ++i) {
            embedding[i] /= tokens.size();
        }
        
        return embedding;
    }
    
    Vector AverageWordEmbedding::generate_embedding(const std::vector<float>& features) {
        if (features.size() != dimension_) {
            throw DimensionMismatchException("Feature dimension mismatch");
        }
        return Vector(features.begin(), features.end());
    }
    
    // Factory function
    std::unique_ptr<EmbeddingGenerator> create_embedding_generator(EmbeddingType type, size_t dimension) {
        switch (type) {
            case EmbeddingType::HASH:
                return std::make_unique<HashEmbedding>(dimension);
            case EmbeddingType::RANDOM_PROJECTION:
                return std::make_unique<RandomProjectionEmbedding>(256, dimension);
            case EmbeddingType::TFIDF:
                return std::make_unique<TFIDFEmbedding>(dimension);
            case EmbeddingType::NGRAM:
                return std::make_unique<NGramEmbedding>(3, dimension);
            case EmbeddingType::AVERAGE_WORD:
                return std::make_unique<AverageWordEmbedding>(dimension);
            default:
                throw std::invalid_argument("Unknown embedding type");
        }
    }
}