// ---------------------------------------------------------------------------
// core/config.hpp -- typed configuration loaded from config/chappie.json.
// ---------------------------------------------------------------------------
#pragma once

#include <string>
#include <vector>

#include "core/json.hpp"

namespace chappie
{

    // ---------------------------------------------------------------------------
    // Typed configuration groups. Each mirrors a section of chappie.json so the
    // rest of the code never touches raw JSON strings.
    // ---------------------------------------------------------------------------
    struct RobotConfig
    {
        std::string name = "Chappie";
        std::string model = "Scout-22";
        std::string serial = "CHP-0001";
        std::string personality_preset = "curious_child";
    };

    struct BrainConfig
    {
        double learning_rate = 0.05;
        std::vector<int> hidden_layers{16, 12};
        int memory_capacity = 5000;
        double decision_threshold = 0.55;
        double curiosity = 0.8;
        int reflection_interval_sec = 30;
    };

    struct PersonalityConfig
    {
        std::string preset = "curious_child";
        double openness = 0.9;
        double conscientiousness = 0.6;
        double extraversion = 0.7;
        double agreeableness = 0.8;
        double neuroticism = 0.3;
        double curiosity = 0.85;
        double playfulness = 0.9;
        double assertiveness = 0.4;
        double mood_baseline = 0.65;
        double mood_volatility = 0.15;
    };

    struct VisionConfig
    {
        bool enabled = true;
        int width = 160;
        int height = 120;
        int channels = 1;
        double detection_threshold = 0.5;
        int frame_history = 60;
    };

    struct SpeechConfig
    {
        bool recognition_enabled = false;
        bool tts_enabled = true;
        std::string language = "en-US";
        std::string voice = "synthetic-young";
        std::string wake_word = "chappie";
        double confidence_threshold = 0.4;
    };

    struct RoboticsConfig
    {
        double max_speed = 1.5;
        double battery_capacity = 100.0;
        double idle_drain_per_sec = 0.01;
        double move_drain_per_sec = 0.08;
        int servo_count = 6;
        double arm_reach_cm = 45.0;
    };

    struct NavigationConfig
    {
        int world_width = 40;
        int world_height = 40;
        double obstacle_density = 0.15;
        std::string algorithm = "astar";
        double step_cost = 1.0;
        bool diagonal = true;
    };

    struct SensorsConfig
    {
        double range_cm = 200.0;
        int beam_count = 8;
        double imu_noise = 0.02;
        double temperature_c = 21.0;
    };

    struct SecurityConfig
    {
        std::string autonomy_level = "supervised";
        std::vector<std::string> require_confirmation;
        std::vector<std::vector<double>> denied_zones; // {x, y, radius}
        double max_speed_without_permission = 0.8;
        bool audit_log = true;
    };

    struct NetworkingConfig
    {
        int port = 8765;
        std::string bind = "127.0.0.1";
        bool enable_remote_control = false;
        int heartbeat_sec = 5;
    };

    struct PluginsConfig
    {
        std::string directory = "plugins";
        bool auto_load = true;
    };

    struct DatabaseConfig
    {
        std::string path = "database/chappie.db";
        int autosave_sec = 10;
    };

    struct LoggingConfig
    {
        std::string level = "info";
        bool console = true;
        std::string file = "logs/chappie.log";
    };

    // ---------------------------------------------------------------------------
    // Config -- the full configuration with load/save and validation.
    // ---------------------------------------------------------------------------
    class Config
    {
    public:
        RobotConfig robot;
        BrainConfig brain;
        PersonalityConfig personality;
        VisionConfig vision;
        SpeechConfig speech;
        RoboticsConfig robotics;
        NavigationConfig navigation;
        SensorsConfig sensors;
        SecurityConfig security;
        NetworkingConfig networking;
        PluginsConfig plugins;
        DatabaseConfig database;
        LoggingConfig logging;

        // Load from a .json file. Returns false and fills `error` on failure.
        bool load(const std::string &path, std::string &error);

        // Save back to disk.
        bool save(const std::string &path) const;

        // Serialize to a Json object (for the GUI / API to read).
        Json to_json() const;

        // Path this config was loaded from (empty if constructed directly).
        const std::string &source_path() const { return source_path_; }

    private:
        std::string source_path_;
    };

} // namespace chappie
