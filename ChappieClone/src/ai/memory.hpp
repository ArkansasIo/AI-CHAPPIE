// ---------------------------------------------------------------------------
// ai/memory.hpp -- persistent, associative memory for the robot.
//
// Three stores working together:
//   * EpisodicMemory  : timestamped events ("what happened")
//   * SemanticMemory  : learned facts (key -> value, confidence)
//   * AssociativeMemory: nearest-neighbour recall over embeddings (hash-based,
//                        since we have no external vector DB)
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/json.hpp"
#include "core/math.hpp"

namespace chappie
{

    // ---------------------------------------------------------------------------
    // A single remembered episode.
    // ---------------------------------------------------------------------------
    struct Episode
    {
        std::uint64_t id = 0;
        std::string timestamp;
        std::string kind; // e.g. "observation", "dialogue", "decision", "event"
        std::string content;
        Vec embedding;         // fixed-size feature vector for similarity search
        double salience = 0.5; // importance 0..1 (affects recall priority)
        int recall_count = 0;  // how often it has been retrieved
    };

    // ---------------------------------------------------------------------------
    // A semantic fact with a confidence value.
    // ---------------------------------------------------------------------------
    struct Fact
    {
        std::string key;
        std::string value;
        double confidence = 1.0;
        std::string source; // "taught", "inferred", "observed"
        std::uint64_t updated_at = 0;
    };

    // ---------------------------------------------------------------------------
    // EpisodicMemory -- bounded, decaying store of episodes.
    // ---------------------------------------------------------------------------
    class EpisodicMemory
    {
    public:
        explicit EpisodicMemory(std::size_t capacity = 5000)
            : capacity_(capacity) {}

        std::uint64_t remember(const std::string &kind, const std::string &content,
                               const Vec &embedding = Vec{}, double salience = 0.5);
        std::uint64_t remember(const Episode &e);

        const std::deque<Episode> &episodes() const { return episodes_; }
        std::size_t size() const { return episodes_.size(); }
        std::size_t capacity() const { return capacity_; }

        // Most recent `n` episodes (optionally filtered by kind).
        std::vector<Episode> recent(std::size_t n,
                                    const std::string &kind = "") const;

        // Retrieval: top-k most similar episodes to a query embedding.
        std::vector<Episode> recall_similar(const Vec &query, std::size_t k = 5) const;

        // Retrieval by substring match on content.
        std::vector<Episode> search(const std::string &needle) const;

        // Retrieve the single most salient episode.
        const Episode *most_salient() const;

        // Forget the lowest-salience fraction (memory consolidation / pruning).
        std::size_t consolidate(double forget_fraction = 0.1);

        void clear()
        {
            episodes_.clear();
            next_id_ = 1;
        }

        Json to_json() const;
        void from_json(const Json &j);

    private:
        std::deque<Episode> episodes_;
        std::size_t capacity_;
        std::uint64_t next_id_ = 1;
    };

    // ---------------------------------------------------------------------------
    // SemanticMemory -- key/value facts with confidence and reinforcement.
    // ---------------------------------------------------------------------------
    class SemanticMemory
    {
    public:
        void learn(const std::string &key, const std::string &value,
                   double confidence = 1.0, const std::string &source = "inferred");
        bool has(const std::string &key) const;
        const Fact *recall(const std::string &key) const;
        std::vector<Fact> all_facts() const;
        std::vector<Fact> search(const std::string &needle) const;

        // Reinforce an existing fact (raises confidence toward 1).
        void reinforce(const std::string &key, double amount = 0.1);

        // Weaken a fact (raises confidence toward 0); erases below threshold.
        void weaken(const std::string &key, double amount = 0.1,
                    double erase_below = 0.05);

        std::size_t size() const { return facts_.size(); }
        void clear() { facts_.clear(); }

        Json to_json() const;
        void from_json(const Json &j);

    private:
        std::map<std::string, Fact> facts_;
        std::uint64_t clock_ = 0;
    };

    // ---------------------------------------------------------------------------
    // AssociativeMemory -- hash-bucketed approximate nearest neighbour recall.
    // Uses random-projection locality-sensitive hashing so recall is O(bucket)
    // instead of O(n), which matters for a real-time robot loop.
    // ---------------------------------------------------------------------------
    class AssociativeMemory
    {
    public:
        AssociativeMemory(std::size_t embedding_dim = 32, std::size_t bits = 8,
                          std::uint32_t seed = 2024);

        void insert(std::uint64_t key, const Vec &embedding);
        void remove(std::uint64_t key);

        // Approximate nearest neighbours (returns episode ids + similarity).
        std::vector<std::pair<std::uint64_t, double>> query(const Vec &embedding,
                                                            std::size_t k = 5) const;

        std::size_t size() const { return buckets_total_; }
        std::size_t embedding_dim() const { return dim_; }

    private:
        std::uint32_t hash_bucket(const Vec &v) const;

        std::size_t dim_;
        std::size_t bits_;
        Vec random_bits_; // dim_ * bits_ hyperplanes
        std::unordered_map<std::uint32_t, std::vector<std::pair<std::uint64_t, Vec>>>
            buckets_;
        std::size_t buckets_total_ = 0;
    };

    // ---------------------------------------------------------------------------
    // Memory -- the facade the Brain talks to.
    // ---------------------------------------------------------------------------
    class Memory
    {
    public:
        Memory(std::size_t capacity = 5000, std::size_t embedding_dim = 32);

        // --- episodic ---
        std::uint64_t remember(const std::string &kind, const std::string &content,
                               const Vec &embedding = Vec{}, double salience = 0.5);
        std::vector<Episode> recall(const Vec &query, std::size_t k = 5) const;
        std::vector<Episode> recent(std::size_t n) const { return episodic_.recent(n); }
        std::vector<Episode> search(const std::string &needle) const
        {
            return episodic_.search(needle);
        }

        // --- semantic ---
        void learn(const std::string &key, const std::string &value,
                   double confidence = 1.0, const std::string &source = "inferred")
        {
            semantic_.learn(key, value, confidence, source);
        }
        const Fact *recall_fact(const std::string &key) const
        {
            return semantic_.recall(key);
        }
        std::vector<Fact> facts() const { return semantic_.all_facts(); }

        // --- statistics ---
        struct Stats
        {
            std::size_t episodes = 0;
            std::size_t facts = 0;
            std::size_t capacity = 0;
            double mean_salience = 0.0;
        };
        Stats stats() const;

        std::size_t episode_count() const { return episodic_.size(); }
        std::size_t fact_count() const { return semantic_.size(); }

        // Consolidation: prune low-salience episodes when over capacity.
        std::size_t consolidate(double forget_fraction = 0.1)
        {
            return episodic_.consolidate(forget_fraction);
        }

        void clear();
        Json to_json() const;
        void from_json(const Json &j);

        const EpisodicMemory &episodic() const { return episodic_; }
        const SemanticMemory &semantic() const { return semantic_; }

    private:
        EpisodicMemory episodic_;
        SemanticMemory semantic_;
        AssociativeMemory associative_;
    };

} // namespace chappie
