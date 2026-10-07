// ---------------------------------------------------------------------------
// core/json.cpp -- JSON parser and serializer implementation.
// ---------------------------------------------------------------------------
#include "core/json.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace chappie
{

    namespace
    {

        // ---------------------------------------------------------------------------
        // Recursive-descent parser state.
        // ---------------------------------------------------------------------------
        struct Parser
        {
            const std::string &s;
            std::size_t i = 0;
            std::string error;

            explicit Parser(const std::string &text) : s(text) {}

            void skip_ws()
            {
                while (i < s.size() &&
                       (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
                    ++i;
            }

            bool fail(const std::string &msg)
            {
                std::ostringstream os;
                os << msg << " at offset " << i;
                error = os.str();
                return false;
            }

            bool parse_value(Json &out)
            {
                skip_ws();
                if (i >= s.size())
                    return fail("unexpected end of input");
                const char c = s[i];
                if (c == '{')
                    return parse_object(out);
                if (c == '[')
                    return parse_array(out);
                if (c == '"')
                {
                    std::string str;
                    if (!parse_string(str))
                        return false;
                    out = Json(str);
                    return true;
                }
                if (c == 't' || c == 'f')
                    return parse_bool(out);
                if (c == 'n')
                    return parse_null(out);
                return parse_number(out);
            }

            bool parse_object(Json &out)
            {
                out = Json::object();
                ++i; // consume '{'
                skip_ws();
                if (i < s.size() && s[i] == '}')
                {
                    ++i;
                    return true;
                }
                while (true)
                {
                    skip_ws();
                    std::string key;
                    if (i >= s.size() || s[i] != '"')
                        return fail("expected string key");
                    if (!parse_string(key))
                        return false;
                    skip_ws();
                    if (i >= s.size() || s[i] != ':')
                        return fail("expected ':' after key");
                    ++i;
                    Json value;
                    if (!parse_value(value))
                        return false;
                    out.set(key, value);
                    skip_ws();
                    if (i >= s.size())
                        return fail("unterminated object");
                    if (s[i] == ',')
                    {
                        ++i;
                        continue;
                    }
                    if (s[i] == '}')
                    {
                        ++i;
                        return true;
                    }
                    return fail("expected ',' or '}'");
                }
            }

            bool parse_array(Json &out)
            {
                out = Json::array();
                ++i; // consume '['
                skip_ws();
                if (i < s.size() && s[i] == ']')
                {
                    ++i;
                    return true;
                }
                while (true)
                {
                    Json value;
                    if (!parse_value(value))
                        return false;
                    out.push_back(value);
                    skip_ws();
                    if (i >= s.size())
                        return fail("unterminated array");
                    if (s[i] == ',')
                    {
                        ++i;
                        continue;
                    }
                    if (s[i] == ']')
                    {
                        ++i;
                        return true;
                    }
                    return fail("expected ',' or ']'");
                }
            }

            bool parse_string(std::string &out)
            {
                ++i; // consume opening quote
                out.clear();
                while (i < s.size())
                {
                    const char c = s[i++];
                    if (c == '"')
                        return true;
                    if (c == '\\')
                    {
                        if (i >= s.size())
                            return fail("bad escape");
                        const char e = s[i++];
                        switch (e)
                        {
                        case '"':
                            out += '"';
                            break;
                        case '\\':
                            out += '\\';
                            break;
                        case '/':
                            out += '/';
                            break;
                        case 'b':
                            out += '\b';
                            break;
                        case 'f':
                            out += '\f';
                            break;
                        case 'n':
                            out += '\n';
                            break;
                        case 'r':
                            out += '\r';
                            break;
                        case 't':
                            out += '\t';
                            break;
                        case 'u':
                        {
                            // \uXXXX -- decode to UTF-8 (BMP only).
                            if (i + 4 > s.size())
                                return fail("bad \\u escape");
                            unsigned code = 0;
                            for (int k = 0; k < 4; ++k)
                            {
                                const char h = s[i++];
                                code <<= 4;
                                if (h >= '0' && h <= '9')
                                    code |= (h - '0');
                                else if (h >= 'a' && h <= 'f')
                                    code |= (h - 'a' + 10);
                                else if (h >= 'A' && h <= 'F')
                                    code |= (h - 'A' + 10);
                                else
                                    return fail("bad hex in \\u escape");
                            }
                            if (code < 0x80)
                            {
                                out += static_cast<char>(code);
                            }
                            else if (code < 0x800)
                            {
                                out += static_cast<char>(0xC0 | (code >> 6));
                                out += static_cast<char>(0x80 | (code & 0x3F));
                            }
                            else
                            {
                                out += static_cast<char>(0xE0 | (code >> 12));
                                out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                                out += static_cast<char>(0x80 | (code & 0x3F));
                            }
                            break;
                        }
                        default:
                            return fail("unknown escape");
                        }
                    }
                    else
                    {
                        out += c;
                    }
                }
                return fail("unterminated string");
            }

            bool parse_bool(Json &out)
            {
                if (s.compare(i, 4, "true") == 0)
                {
                    i += 4;
                    out = Json(true);
                    return true;
                }
                if (s.compare(i, 5, "false") == 0)
                {
                    i += 5;
                    out = Json(false);
                    return true;
                }
                return fail("invalid literal");
            }

            bool parse_null(Json &out)
            {
                if (s.compare(i, 4, "null") == 0)
                {
                    i += 4;
                    out = Json();
                    return true;
                }
                return fail("invalid literal");
            }

            bool parse_number(Json &out)
            {
                const std::size_t start = i;
                if (i < s.size() && (s[i] == '-' || s[i] == '+'))
                    ++i;
                while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
                    ++i;
                if (i < s.size() && s[i] == '.')
                {
                    ++i;
                    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
                        ++i;
                }
                if (i < s.size() && (s[i] == 'e' || s[i] == 'E'))
                {
                    ++i;
                    if (i < s.size() && (s[i] == '-' || s[i] == '+'))
                        ++i;
                    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
                        ++i;
                }
                if (i == start)
                    return fail("invalid number");
                try
                {
                    out = Json(std::stod(s.substr(start, i - start)));
                }
                catch (const std::exception &)
                {
                    return fail("number out of range");
                }
                return true;
            }
        };

        void escape_to(const std::string &in, std::string &out)
        {
            out += '"';
            for (char c : in)
            {
                switch (c)
                {
                case '"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                case '\b':
                    out += "\\b";
                    break;
                case '\f':
                    out += "\\f";
                    break;
                default:
                    out += c;
                }
            }
            out += '"';
        }

    } // namespace

    // ---------------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------------
    const Json &Json::operator[](const std::string &key) const
    {
        static const Json null_value;
        if (type_ != Type::Object)
            return null_value;
        auto it = obj_.find(key);
        return it == obj_.end() ? null_value : it->second;
    }

    Json &Json::operator[](const std::string &key)
    {
        if (type_ != Type::Object)
        {
            type_ = Type::Object;
            obj_.clear();
        }
        return obj_[key];
    }

    const Json &Json::operator[](std::size_t idx) const
    {
        static const Json null_value;
        if (type_ != Type::Array || idx >= arr_.size())
            return null_value;
        return arr_[idx];
    }

    void Json::push_back(const Json &v)
    {
        if (type_ != Type::Array)
        {
            type_ = Type::Array;
            arr_.clear();
        }
        arr_.push_back(v);
    }

    const Json &Json::lookup_path(const std::string &path) const
    {
        const Json *node = this;
        std::size_t start = 0;
        while (start <= path.size())
        {
            const std::size_t dot = path.find('.', start);
            const std::string key = path.substr(
                start, dot == std::string::npos ? std::string::npos : dot - start);
            node = &(*node)[key];
            if (dot == std::string::npos)
                break;
            start = dot + 1;
        }
        return *node;
    }

    bool Json::get_bool(const std::string &path, bool def) const
    {
        return lookup_path(path).as_bool(def);
    }
    double Json::get_number(const std::string &path, double def) const
    {
        return lookup_path(path).as_number(def);
    }
    int Json::get_int(const std::string &path, int def) const
    {
        return lookup_path(path).as_int(def);
    }
    std::string Json::get_string(const std::string &path,
                                 const std::string &def) const
    {
        return lookup_path(path).as_string(def);
    }

    std::vector<double> Json::get_number_array(const std::string &path) const
    {
        std::vector<double> out;
        const Json &node = lookup_path(path);
        if (node.is_array())
            for (std::size_t i = 0; i < node.size(); ++i)
                out.push_back(node[i].as_number());
        return out;
    }

    // ---------------------------------------------------------------------------
    // Serialization
    // ---------------------------------------------------------------------------
    void Json::dump_to(std::string &out, int indent, int depth) const
    {
        const bool pretty = indent > 0;
        const std::string pad = pretty ? std::string((depth + 1) * indent, ' ') : "";
        const std::string pad_end = pretty ? std::string(depth * indent, ' ') : "";

        switch (type_)
        {
        case Type::Null:
            out += "null";
            break;
        case Type::Bool:
            out += bool_ ? "true" : "false";
            break;
        case Type::Number:
        {
            std::ostringstream os;
            // Print integers without a trailing ".0".
            if (num_ == static_cast<long long>(num_))
                os << static_cast<long long>(num_);
            else
                os << num_;
            out += os.str();
            break;
        }
        case Type::String:
            escape_to(str_, out);
            break;
        case Type::Array:
        {
            if (arr_.empty())
            {
                out += "[]";
                break;
            }
            out += "[";
            for (std::size_t i = 0; i < arr_.size(); ++i)
            {
                if (pretty)
                {
                    out += "\n" + pad;
                }
                else if (i > 0)
                {
                    out += ",";
                }
                if (!pretty && i > 0)
                    out += "";
                arr_[i].dump_to(out, indent, depth + 1);
            }
            if (pretty)
                out += "\n" + pad_end;
            out += "]";
            break;
        }
        case Type::Object:
        {
            if (obj_.empty())
            {
                out += "{}";
                break;
            }
            out += "{";
            std::size_t n = 0;
            for (const auto &kv : obj_)
            {
                if (pretty)
                {
                    out += "\n" + pad;
                }
                else if (n > 0)
                {
                    out += ",";
                }
                escape_to(kv.first, out);
                out += pretty ? ": " : ":";
                kv.second.dump_to(out, indent, depth + 1);
                ++n;
            }
            if (pretty)
                out += "\n" + pad_end;
            out += "}";
            break;
        }
        }
    }

    std::string Json::dump(int indent) const
    {
        std::string out;
        dump_to(out, indent, 0);
        return out;
    }

    // ---------------------------------------------------------------------------
    // Parsing entry points
    // ---------------------------------------------------------------------------
    bool Json::parse(const std::string &text, Json &out, std::string &error)
    {
        Parser p(text);
        if (!p.parse_value(out))
        {
            error = p.error;
            return false;
        }
        p.skip_ws();
        if (p.i != text.size())
        {
            error = "trailing characters after JSON value";
            return false;
        }
        error.clear();
        return true;
    }

    bool Json::load_file(const std::string &path, Json &out, std::string &error)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            error = "cannot open file: " + path;
            return false;
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        return parse(ss.str(), out, error);
    }

    bool Json::save_file(const std::string &path) const
    {
        std::ofstream out(path, std::ios::binary);
        if (!out)
            return false;
        out << dump(2) << "\n";
        return static_cast<bool>(out);
    }

    Json Json::parse_or_throw(const std::string &text)
    {
        Json j;
        std::string err;
        if (!parse(text, j, err))
            throw std::runtime_error("JSON parse error: " + err);
        return j;
    }

} // namespace chappie
