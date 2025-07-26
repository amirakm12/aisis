#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <unordered_set>

namespace VectorDB {
    // Abstract base class for embedding generators
    class EmbeddingGenerator {
    public:
        virtual ~EmbeddingGenerator() = default;
        virtual Vector generate_embedding(const std::string& text) = 0;
        virtual Vector generate_embedding(const std::vector<float>& features) = 0;
        virtual size_t get_dimension() const = 0;
    };
    
    // Simple hash-based embedding (for demonstration)
    class HashEmbedding : public EmbeddingGenerator {
    private:
        size_t dimension_;
        std::hash<std::string> hasher_;
        std::mt19937 rng_;
        
    public:
        explicit HashEmbedding(size_t dimension = 128);
        
        Vector generate_embedding(const std::string& text) override;
        Vector generate_embedding(const std::vector<float>& features) override;
        size_t get_dimension() const override { return dimension_; }
    };
    
    // Random projection embedding
    class RandomProjectionEmbedding : public EmbeddingGenerator {
    private:
        size_t input_dimension_;
        size_t output_dimension_;
        std::vector<std::vector<float>> projection_matrix_;
        std::mt19937 rng_;
        
        void initialize_projection_matrix();
        
    public:
        RandomProjectionEmbedding(size_t input_dim, size_t output_dim = 128);
        
        Vector generate_embedding(const std::string& text) override;
        Vector generate_embedding(const std::vector<float>& features) override;
        size_t get_dimension() const override { return output_dimension_; }
    };
    
    // TF-IDF based embedding
    class TFIDFEmbedding : public EmbeddingGenerator {
    private:
        std::unordered_map<std::string, size_t> vocabulary_;
        std::unordered_map<std::string, float> idf_scores_;
        size_t max_features_;
        bool is_fitted_;
        
        std::vector<std::string> tokenize(const std::string& text) const;
        void build_vocabulary(const std::vector<std::string>& documents);
        void calculate_idf(const std::vector<std::string>& documents);
        
    public:
        explicit TFIDFEmbedding(size_t max_features = 1000);
        
        void fit(const std::vector<std::string>& documents);
        Vector generate_embedding(const std::string& text) override;
        Vector generate_embedding(const std::vector<float>& features) override;
        size_t get_dimension() const override { return max_features_; }
        
        bool is_fitted() const { return is_fitted_; }
    };
    
    // N-gram based embedding
    class NGramEmbedding : public EmbeddingGenerator {
    private:
        size_t n_;
        size_t dimension_;
        std::unordered_map<std::string, size_t> ngram_to_index_;
        std::hash<std::string> hasher_;
        
        std::vector<std::string> generate_ngrams(const std::string& text) const;
        
    public:
        NGramEmbedding(size_t n = 3, size_t dimension = 128);
        
        Vector generate_embedding(const std::string& text) override;
        Vector generate_embedding(const std::vector<float>& features) override;
        size_t get_dimension() const override { return dimension_; }
    };
    
    // Sentence embedding using simple averaging
    class AverageWordEmbedding : public EmbeddingGenerator {
    private:
        std::unordered_map<std::string, Vector> word_embeddings_;
        size_t dimension_;
        std::mt19937 rng_;
        
        std::vector<std::string> tokenize(const std::string& text) const;
        Vector get_word_embedding(const std::string& word);
        
    public:
        explicit AverageWordEmbedding(size_t dimension = 128);
        
        void add_word_embedding(const std::string& word, const Vector& embedding);
        void load_word_embeddings(const std::string& file_path);
        
        Vector generate_embedding(const std::string& text) override;
        Vector generate_embedding(const std::vector<float>& features) override;
        size_t get_dimension() const override { return dimension_; }
    };
    
    // Factory function to create embedding generators
    enum class EmbeddingType {
        HASH,
        RANDOM_PROJECTION,
        TFIDF,
        NGRAM,
        AVERAGE_WORD
    };
    
    std::unique_ptr<EmbeddingGenerator> create_embedding_generator(
        EmbeddingType type, size_t dimension = 128);
}