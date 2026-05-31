#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace neurograph::training {

/// Configuration for embedding categorical features
struct EmbeddingConfig {
    bool enabled = false;
    int embed_dim = 16;                        // dimensions per embedding
    std::vector<int> categorical_columns;      // which feature columns are categorical (by index)
};

/// Learnable lookup table that maps discrete token/category IDs to dense vectors.
///
/// Weights are initialised with Xavier uniform and can be updated via SGD-style
/// gradient steps.  This is the same primitive used for word embeddings in NLP
/// and for categorical-feature embeddings in tabular learning.
class EmbeddingTable {
public:
    /// @param vocab_size  number of unique tokens / categories
    /// @param embed_dim   dimensionality of each embedding vector
    EmbeddingTable(int vocab_size, int embed_dim)
        : vocab_size_(vocab_size), embed_dim_(embed_dim)
    {
        // Xavier uniform: U(-limit, +limit) where limit = sqrt(6 / (fan_in + fan_out))
        const double limit = std::sqrt(6.0 / (vocab_size_ + embed_dim_));

        thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<double> dist(-limit, limit);

        weights_.resize(vocab_size_);
        for (auto& row : weights_) {
            row.resize(embed_dim_);
            for (auto& v : row)
                v = dist(rng);
        }
    }

    /// Look up the embedding vector for @p token_id.
    /// Returns a zero vector if the id is out of range.
    std::vector<double> lookup(int token_id) const {
        if (token_id < 0 || token_id >= vocab_size_)
            return std::vector<double>(embed_dim_, 0.0);
        return weights_[token_id];
    }

    /// Mutable access to the full weight matrix (vocab_size x embed_dim).
    std::vector<std::vector<double>>& weights() { return weights_; }

    /// Const access to the full weight matrix.
    const std::vector<std::vector<double>>& weights() const { return weights_; }

    /// SGD-style in-place update:  w[token_id][j] -= lr * gradient[j]
    void update(int token_id, const std::vector<double>& gradient, double lr) {
        if (token_id < 0 || token_id >= vocab_size_)
            return;
        const int dims = std::min(embed_dim_, static_cast<int>(gradient.size()));
        for (int j = 0; j < dims; ++j)
            weights_[token_id][j] -= lr * gradient[j];
    }

    int vocab_size() const { return vocab_size_; }
    int embed_dim()  const { return embed_dim_; }

private:
    int vocab_size_;
    int embed_dim_;
    std::vector<std::vector<double>> weights_;   // [vocab_size][embed_dim]
};

} // namespace neurograph::training
