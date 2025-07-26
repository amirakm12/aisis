#include "similarity.h"
#include <stdexcept>
#include <numeric>

namespace VectorDB {
    
    float SimilarityCalculator::calculate_distance(const Vector& a, const Vector& b, DistanceMetric metric) {
        switch (metric) {
            case DistanceMetric::EUCLIDEAN:
                return euclidean_distance(a, b);
            case DistanceMetric::COSINE:
                return cosine_distance(a, b);
            case DistanceMetric::DOT_PRODUCT:
                return -dot_product(a, b); // Negative for similarity to distance conversion
            case DistanceMetric::MANHATTAN:
                return manhattan_distance(a, b);
            default:
                throw std::invalid_argument("Unknown distance metric");
        }
    }
    
    float SimilarityCalculator::euclidean_distance(const Vector& a, const Vector& b) {
        validate_dimensions(a, b);
        float sum = 0.0f;
        
#ifdef USE_OPENMP
        #pragma omp simd reduction(+:sum)
#endif
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    float SimilarityCalculator::cosine_distance(const Vector& a, const Vector& b) {
        validate_dimensions(a, b);
        
        float dot_prod = dot_product(a, b);
        float mag_a = vector_magnitude(a);
        float mag_b = vector_magnitude(b);
        
        if (mag_a == 0.0f || mag_b == 0.0f) {
            return 1.0f; // Maximum distance for zero vectors
        }
        
        float cosine_sim = dot_prod / (mag_a * mag_b);
        // Clamp to [-1, 1] to handle floating point errors
        cosine_sim = std::max(-1.0f, std::min(1.0f, cosine_sim));
        
        // Convert similarity to distance: distance = 1 - similarity
        return 1.0f - cosine_sim;
    }
    
    float SimilarityCalculator::manhattan_distance(const Vector& a, const Vector& b) {
        validate_dimensions(a, b);
        float sum = 0.0f;
        
#ifdef USE_OPENMP
        #pragma omp simd reduction(+:sum)
#endif
        for (size_t i = 0; i < a.size(); ++i) {
            sum += std::abs(a[i] - b[i]);
        }
        return sum;
    }
    
    Vector SimilarityCalculator::normalize_vector(const Vector& vec) {
        float magnitude = vector_magnitude(vec);
        if (magnitude == 0.0f) {
            return vec; // Return zero vector as is
        }
        
        Vector normalized(vec.size());
#ifdef USE_OPENMP
        #pragma omp simd
#endif
        for (size_t i = 0; i < vec.size(); ++i) {
            normalized[i] = vec[i] / magnitude;
        }
        return normalized;
    }
    
    float SimilarityCalculator::cosine_similarity(const Vector& a, const Vector& b) {
        validate_dimensions(a, b);
        
        float dot_prod = dot_product(a, b);
        float mag_a = vector_magnitude(a);
        float mag_b = vector_magnitude(b);
        
        if (mag_a == 0.0f || mag_b == 0.0f) {
            return 0.0f; // No similarity for zero vectors
        }
        
        return dot_prod / (mag_a * mag_b);
    }
    
    void SimilarityCalculator::validate_dimensions(const Vector& a, const Vector& b) {
        if (a.size() != b.size()) {
            throw DimensionMismatchException(
                "Vector dimensions don't match: " + std::to_string(a.size()) + 
                " vs " + std::to_string(b.size())
            );
        }
    }
}