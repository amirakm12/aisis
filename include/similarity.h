#pragma once

#include "types.h"
#include <cmath>
#include <algorithm>

namespace VectorDB {
    class SimilarityCalculator {
    public:
        static float calculate_distance(const Vector& a, const Vector& b, DistanceMetric metric);
        
        // Specific distance functions
        static float euclidean_distance(const Vector& a, const Vector& b);
        static float cosine_distance(const Vector& a, const Vector& b);
        static float dot_product(const Vector& a, const Vector& b);
        static float manhattan_distance(const Vector& a, const Vector& b);
        
        // Utility functions
        static float vector_magnitude(const Vector& vec);
        static Vector normalize_vector(const Vector& vec);
        static float cosine_similarity(const Vector& a, const Vector& b);
        
    private:
        static void validate_dimensions(const Vector& a, const Vector& b);
    };
    
    // Inline implementations for performance-critical functions
    inline float SimilarityCalculator::dot_product(const Vector& a, const Vector& b) {
        validate_dimensions(a, b);
        float result = 0.0f;
        
#ifdef USE_OPENMP
        #pragma omp simd reduction(+:result)
#endif
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    inline float SimilarityCalculator::vector_magnitude(const Vector& vec) {
        float sum = 0.0f;
        
#ifdef USE_OPENMP
        #pragma omp simd reduction(+:sum)
#endif
        for (float val : vec) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }
}