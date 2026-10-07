// ---------------------------------------------------------------------------
// ai/memory.cpp -- episodic, semantic and associative memory implementation.
// ---------------------------------------------------------------------------
#include "ai/memory.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "core/logger.hpp"

namespace chappie
{

    namespace
    {

        std::string now_string()
        {
            const std::time_t t = std::time(nullptr);
            std::tm tm_buf{};
#if defined(_WIN32)
            localtime_s(&tm_buf, &t);
#else
            localtime_r(&t, &tm_buf);
#endif
            std::ostringstream os;
            os << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
            return os.str();
        }

        double cosine(const Vec &a, const Vec &b)
        {
            if (a.size() != b.size() || a.empty())
                return 0.0;
            const double na = a.norm();
            const double nb = b.norm();
            if (na <= 0.0 || nb <= 0.0)
                return 0.0;
            return a.dot(b) / (na * nb);
        }

    } // namespace

    // ===========================================================================
    // EpisodicMemory
    // ===========================================================================
    std::uint64_t EpisodicMemory::remember(const std::string &kind,
                                           const std::string &content,
                                           const Vec &embedding, double salience)
    {
        Episode e;
        e.id = next_id_++;
        e.timestamp = now_string();
        e.kind = kind;
        e.content = content;
        e.embedding = embedding;
        e.salience = clamp(salience, 0.0, 1.0);
        return remember(e);
    }

    std::uint64_t EpisodicMemory::remember(const Episode &e)
    {
        Episode copy = e;
        if (copy.id == 0)
            copy.id = next_id_++;
        if (copy.timestamp.empty())
            copy.timestamp = now_string();

        episodes_.push_back(copy);

        // Enforce capacity by dropping the least salient episodes.
        while (episodes_.size() > capacity_)
        {
            auto weakest = std::min_element(
                episodes_.begin(), episodes_.end(),
                [](const Episode &a, const Episode &b)
                {
                    return a.salience < b.salience;
                });
            if (weakest == episodes_.end())
                break;
            episodes_.erase(weakest);
        }
        return copy.id;
    }

    std::vector<Episode> EpisodicMemory::recent(std::size_t n,
                                                const std::string &kind) const
    {
        std::vector<Episode> out;
        for (auto it = episodes_.rbegin(); it != episodes_.rend(); ++it)
        {
            if (!kind.empty() && it->kind != kind)
                continue;
            out.push_back(*it);
            if (out.size() >= n)
                break;
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    std::vector<Episode> EpisodicMemory::recall_similar(const Vec &query,
                                                        std::size_t k) const
    {
        std::vector<std::pair<double, const Episode *>> scored;
        for (const auto &e : episodes_)
        {
            if (e.embedding.empty() || e.embedding.size() != query.size())
                continue;
            // Similarity blended with salience so important memories surface.
            const double sim = cosine(e.embedding, query) + 0.25 * e.salience;
            scored.emplace_back(sim, &e);
        }
        const std::size_t take = std::min(k, scored.size());
        std::partial_sort(scored.begin(), scored.begin() + take, scored.end(),
                          [](const auto &a, const auto &b)
                          { return a.first > b.first; });
        std::vector<Episode> out;
        for (std::size_t i = 0; i < take; ++i)
            out.push_back(*scored[i].second);
        return out;
    }

    std::vector<Episode> EpisodicMemory::search(const std::string &needle) const
    {
        std::vector<Episode> out;
        if (needle.empty())
            return out;
        for (const auto &e : episodes_)
            if (e.content.find(needle) != std::string::npos)
                out.push_back(e);
        return out;
    }

    const Episode *EpisodicMemory::most_salient() const
    {
        if (episodes_.empty())
            return nullptr;
        return &*std::max_element(
            episodes_.begin(), episodes_.end(),
            [](const Episode &a, const Episode &b)
            { return a.salience < b.salience; });
    }

    std::size_t EpisodicMemory::consolidate(double forget_fraction)
    {
        if (episodes_.empty())
            return 0;
        const double f = clamp(forget_fraction, 0.0, 0.9);
        std::size_t to_forget =
            static_cast<std::size_t>(f * static_cast<double>(episodes_.size()));
        if (to_forget == 0)
            return 0;

        std::vector<std::size_t> idx(episodes_.size());
        for (std::size_t i = 0; i < idx.size(); ++i)
            idx[i] = i;

        // Sort indices by (salience asc, recency desc) -> forget weakest first.
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b)
                         {
        const double sa = episodes_[a].salience - 0.001 * episodes_[a].recall_count;
        const double sb = episodes_[b].salience - 0.001 * episodes_[b].recall_count;
        return sa < sb; });

        std::vector<bool> drop(episodes_.size(), false);
        for (std::size_t i = 0; i < to_forget; ++i)
            drop[idx[i]] = true;

        std::deque<Episode> kept;
        for (std::size_t i = 0; i < episodes_.size(); ++i)
            if (!drop[i])
                kept.push_back(episodes_[i]);
        const std::size_t removed = episodes_.size() - kept.size();
        episodes_.swap(kept);
        return removed;
    }

    Json EpisodicMemory::to_json() const
    {
        Json arr = Json::array();
        for (const auto &e : episodes_)
        {
            Json j = Json::object();
            j.set("id", Json(static_cast<double>(e.id)));
            j.set("timestamp", Json(e.timestamp));
            j.set("kind", Json(e.kind));
            j.set("content", Json(e.content));
            j.set("salience", Json(e.salience));
            j.set("recall_count", Json(e.recall_count));
            Json emb = Json::array();
            for (double v : e.embedding.raw())
                emb.push_back(Json(v));
            j.set("embedding", emb);
            arr.push_back(j);
        }
        return arr;
    }

    void EpisodicMemory::from_json(const Json &j)
    {
        episodes_.clear();
        next_id_ = 1;
        for (std::size_t i = 0; i < j.size(); ++i)
        {
            const Json &item = j[i];
            Episode e;
            e.id = static_cast<std::uint64_t>(item.get_number("id", 0.0));
            e.timestamp = item.get_string("timestamp", now_string());
            e.kind = item.get_string("kind", "event");
            e.content = item.get_string("content", "");
            e.salience = item.get_number("salience", 0.5);
            e.recall_count = item.get_int("recall_count", 0);
            const Json &emb = item["embedding"];
            std::vector<double> vals;
            for (std::size_t k = 0; k < emb.size(); ++k)
                vals.push_back(emb[k].as_number());
            e.embedding = Vec(vals);
            if (e.id == 0)
                e.id = next_id_;
            next_id_ = std::max(next_id_, e.id + 1);
            episodes_.push_back(e);
        }
    }

    // ===========================================================================
    // SemanticMemory
    // ===========================================================================
    void SemanticMemory::learn(const std::string &key, const std::string &value,
                               double confidence, const std::string &source)
    {
        auto it = facts_.find(key);
        if (it == facts_.end())
        {
            Fact f;
            f.key = key;
            f.value = value;
            f.confidence = clamp(confidence, 0.0, 1.0);
            f.source = source;
            f.updated_at = ++clock_;
            facts_[key] = f;
        }
        else
        {
            // Conflicting evidence blends: new value wins if more confident.
            if (confidence >= it->second.confidence)
            {
                it->second.value = value;
                it->second.source = source;
            }
            // Repeated learning always raises confidence (evidence accumulation).
            it->second.confidence =
                clamp(std::max(it->second.confidence, confidence) + 0.05, 0.0, 1.0);
            it->second.updated_at = ++clock_;
        }
    }

    bool SemanticMemory::has(const std::string &key) const
    {
        return facts_.count(key) > 0;
    }

    const Fact *SemanticMemory::recall(const std::string &key) const
    {
        auto it = facts_.find(key);
        return it == facts_.end() ? nullptr : &it->second;
    }

    std::vector<Fact> SemanticMemory::all_facts() const
    {
        std::vector<Fact> out;
        out.reserve(facts_.size());
        for (const auto &kv : facts_)
            out.push_back(kv.second);
        return out;
    }

    std::vector<Fact> SemanticMemory::search(const std::string &needle) const
    {
        std::vector<Fact> out;
        if (needle.empty())
            return out;
        for (const auto &kv : facts_)
        {
            const Fact &f = kv.second;
            if (f.key.find(needle) != std::string::npos ||
                f.value.find(needle) != std::string::npos)
                out.push_back(f);
        }
        return out;
    }

    void SemanticMemory::reinforce(const std::string &key, double amount)
    {
        auto it = facts_.find(key);
        if (it == facts_.end())
            return;
        it->second.confidence = clamp(it->second.confidence + amount, 0.0, 1.0);
        it->second.updated_at = ++clock_;
    }

    void SemanticMemory::weaken(const std::string &key, double amount,
                                double erase_below)
    {
        auto it = facts_.find(key);
        if (it == facts_.end())
            return;
        it->second.confidence = clamp(it->second.confidence - amount, 0.0, 1.0);
        it->second.updated_at = ++clock_;
        if (it->second.confidence < erase_below)
            facts_.erase(it);
    }

    Json SemanticMemory::to_json() const
    {
        Json arr = Json::array();
        for (const auto &kv : facts_)
        {
            Json j = Json::object();
            j.set("key", Json(kv.second.key));
            j.set("value", Json(kv.second.value));
            j.set("confidence", Json(kv.second.confidence));
            j.set("source", Json(kv.second.source));
            j.set("updated_at", Json(static_cast<double>(kv.second.updated_at)));
            arr.push_back(j);
        }
        return arr;
    }

    void SemanticMemory::from_json(const Json &j)
    {
        facts_.clear();
        clock_ = 0;
        for (std::size_t i = 0; i < j.size(); ++i)
        {
            const Json &item = j[i];
            Fact f;
            f.key = item.get_string("key", "");
            f.value = item.get_string("value", "");
            f.confidence = item.get_number("confidence", 1.0);
            f.source = item.get_string("source", "inferred");
            f.updated_at = static_cast<std::uint64_t>(item.get_number("updated_at", 0.0));
            clock_ = std::max(clock_, f.updated_at);
            if (!f.key.empty())
                facts_[f.key] = f;
        }
    }

    // ===========================================================================
    // AssociativeMemory (random-projection LSH)
    // ===========================================================================
    AssociativeMemory::AssociativeMemory(std::size_t embedding_dim, std::size_t bits,
                                         std::uint32_t seed)
        : dim_(embedding_dim), bits_(bits)
    {
        // bits_ * dim_ random hyperplanes for sign hashing.
        Vec rb(dim_ * bits_);
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (std::size_t i = 0; i < rb.size(); ++i)
            rb[i] = dist(rng);
        random_bits_ = rb;
    }

    std::uint32_t AssociativeMemory::hash_bucket(const Vec &v) const
    {
        std::uint32_t bucket = 0;
        for (std::size_t b = 0; b < bits_; ++b)
        {
            double acc = 0.0;
            for (std::size_t d = 0; d < dim_ && d < v.size(); ++d)
                acc += v[d] * random_bits_[b * dim_ + d];
            if (acc >= 0.0)
                bucket |= (1u << b);
        }
        return bucket;
    }

    void AssociativeMemory::insert(std::uint64_t key, const Vec &embedding)
    {
        Vec padded(dim_, 0.0);
        for (std::size_t i = 0; i < dim_ && i < embedding.size(); ++i)
            padded[i] = embedding[i];
        const std::uint32_t b = hash_bucket(padded);
        buckets_[b].emplace_back(key, padded);
        ++buckets_total_;
    }

    void AssociativeMemory::remove(std::uint64_t key)
    {
        for (auto &kv : buckets_)
        {
            auto &vec = kv.second;
            const std::size_t before = vec.size();
            vec.erase(std::remove_if(vec.begin(), vec.end(),
                                     [key](const auto &p)
                                     { return p.first == key; }),
                      vec.end());
            buckets_total_ -= (before - vec.size());
        }
    }

    std::vector<std::pair<std::uint64_t, double>> AssociativeMemory::query(
        const Vec &embedding, std::size_t k) const
    {
        Vec padded(dim_, 0.0);
        for (std::size_t i = 0; i < dim_ && i < embedding.size(); ++i)
            padded[i] = embedding[i];

        // Gather the exact bucket plus near neighbours (single-bit flips).
        const std::uint32_t base = hash_bucket(padded);
        std::vector<std::pair<std::uint64_t, double>> out;
        auto consider = [&](std::uint32_t bucket)
        {
            auto it = buckets_.find(bucket);
            if (it == buckets_.end())
                return;
            for (const auto &p : it->second)
                out.emplace_back(p.first, cosine(p.second, padded));
        };
        consider(base);
        for (std::size_t b = 0; b < bits_; ++b)
            consider(base ^ (1u << b));

        const std::size_t take = std::min(k, out.size());
        std::partial_sort(out.begin(), out.begin() + take, out.end(),
                          [](const auto &a, const auto &b)
                          { return a.second > b.second; });
        out.resize(take);
        return out;
    }

    // ===========================================================================
    // Memory facade
    // ===========================================================================
    Memory::Memory(std::size_t capacity, std::size_t embedding_dim)
        : episodic_(capacity), associative_(embedding_dim, 8, 2024) {}

    std::uint64_t Memory::remember(const std::string &kind,
                                   const std::string &content,
                                   const Vec &embedding, double salience)
    {
        const std::uint64_t id = episodic_.remember(kind, content, embedding, salience);
        if (!embedding.empty())
            associative_.insert(id, embedding);
        return id;
    }

    std::vector<Episode> Memory::recall(const Vec &query, std::size_t k) const
    {
        // Prefer the LSH index; fall back to exhaustive search when the query has
        // no usable embedding.
        if (query.empty())
            return {};

        const auto candidates = associative_.query(query, k * 3);
        if (candidates.empty())
            return episodic_.recall_similar(query, k);

        std::map<std::uint64_t, double> score;
        for (const auto &c : candidates)
            score[c.first] = c.second;

        std::vector<Episode> out;
        for (const auto &e : episodic_.episodes())
        {
            auto it = score.find(e.id);
            if (it != score.end())
            {
                Episode copy = e;
                copy.recall_count = e.recall_count + 1;
                out.push_back(copy);
            }
        }
        if (out.empty())
            return episodic_.recall_similar(query, k);

        std::sort(out.begin(), out.end(), [&](const Episode &a, const Episode &b)
                  { return score[a.id] > score[b.id]; });
        if (out.size() > k)
            out.resize(k);
        return out;
    }

    Memory::Stats Memory::stats() const
    {
        Stats s;
        s.episodes = episodic_.size();
        s.facts = semantic_.size();
        s.capacity = episodic_.capacity();
        if (!episodic_.episodes().empty())
        {
            double total = 0.0;
            for (const auto &e : episodic_.episodes())
                total += e.salience;
            s.mean_salience = total / static_cast<double>(episodic_.episodes().size());
        }
        return s;
    }

    void Memory::clear()
    {
        episodic_.clear();
        semantic_.clear();
    }

    Json Memory::to_json() const
    {
        Json j = Json::object();
        j.set("episodes", episodic_.to_json());
        j.set("facts", semantic_.to_json());
        return j;
    }

    void Memory::from_json(const Json &j)
    {
        episodic_.from_json(j["episodes"]);
        semantic_.from_json(j["facts"]);
        // Rebuild the associative index.
        associative_ = AssociativeMemory(32, 8, 2024);
        for (const auto &e : episodic_.episodes())
            if (!e.embedding.empty())
                associative_.insert(e.id, e.embedding);
    }

} // namespace chappie
