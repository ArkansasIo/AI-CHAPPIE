// ---------------------------------------------------------------------------
// core/json.hpp -- a small, dependency-free JSON parser/serializer.
//
// Used for config files (config/chappie.json), plugin manifests, network
// messages, and database exports. Supports the full JSON grammar:
// objects, arrays, strings (with escapes), numbers, true/false/null.
// ---------------------------------------------------------------------------
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace chappie
{

    class Json
    {
    public:
        enum class Type
        {
            Null,
            Bool,
            Number,
            String,
            Array,
            Object
        };

        Json() : type_(Type::Null) {}
        Json(bool b) : type_(Type::Bool), bool_(b) {}
        Json(double n) : type_(Type::Number), num_(n) {}
        Json(int n) : type_(Type::Number), num_(static_cast<double>(n)) {}
        Json(const char *s) : type_(Type::String), str_(s) {}
        Json(const std::string &s) : type_(Type::String), str_(s) {}

        static Json array()
        {
            Json j;
            j.type_ = Type::Array;
            return j;
        }
        static Json object()
        {
            Json j;
            j.type_ = Type::Object;
            return j;
        }

        Type type() const { return type_; }
        bool is_null() const { return type_ == Type::Null; }
        bool is_bool() const { return type_ == Type::Bool; }
        bool is_number() const { return type_ == Type::Number; }
        bool is_string() const { return type_ == Type::String; }
        bool is_array() const { return type_ == Type::Array; }
        bool is_object() const { return type_ == Type::Object; }

        // -- accessors with defaults ------------------------------------------
        bool as_bool(bool def = false) const
        {
            return type_ == Type::Bool ? bool_ : def;
        }
        double as_number(double def = 0.0) const
        {
            return type_ == Type::Number ? num_ : def;
        }
        int as_int(int def = 0) const
        {
            return type_ == Type::Number ? static_cast<int>(num_) : def;
        }
        std::string as_string(const std::string &def = "") const
        {
            return type_ == Type::String ? str_ : def;
        }

        // -- object access ----------------------------------------------------
        bool has(const std::string &key) const
        {
            return type_ == Type::Object && obj_.count(key) > 0;
        }
        const Json &operator[](const std::string &key) const;
        Json &operator[](const std::string &key);

        // Typed, nested lookups with defaults (e.g. cfg.get("robot.name", "?")).
        bool get_bool(const std::string &path, bool def = false) const;
        double get_number(const std::string &path, double def = 0.0) const;
        int get_int(const std::string &path, int def = 0) const;
        std::string get_string(const std::string &path,
                               const std::string &def = "") const;
        std::vector<double> get_number_array(const std::string &path) const;

        // -- array access -----------------------------------------------------
        std::size_t size() const
        {
            if (type_ == Type::Array)
                return arr_.size();
            if (type_ == Type::Object)
                return obj_.size();
            return 0;
        }
        const Json &operator[](std::size_t i) const;
        void push_back(const Json &v);

        // -- mutation ---------------------------------------------------------
        void set(const std::string &key, const Json &value) { obj_[key] = value; }

        // -- serialization ----------------------------------------------------
        std::string dump(int indent = 0) const;

        // -- parsing ----------------------------------------------------------
        // Returns true on success; on failure `error` describes the problem.
        static bool parse(const std::string &text, Json &out, std::string &error);

        // Reads a .json file; returns false and fills `error` on failure.
        static bool load_file(const std::string &path, Json &out,
                              std::string &error);
        bool save_file(const std::string &path) const;

        // Convenience: parse and throw std::runtime_error on failure.
        static Json parse_or_throw(const std::string &text);

    private:
        Type type_;
        bool bool_ = false;
        double num_ = 0.0;
        std::string str_;
        std::vector<Json> arr_;
        std::map<std::string, Json> obj_;

        const Json &lookup_path(const std::string &path) const;
        void dump_to(std::string &out, int indent, int depth) const;
    };

} // namespace chappie
