// ---------------------------------------------------------------------------
// core/config.cpp -- load/save/validate configuration.
// ---------------------------------------------------------------------------
#include "core/config.hpp"

#include <stdexcept>

namespace chappie
{

    namespace
    {

        std::vector<std::vector<double>> parse_denied_zones(const Json &arr)
        {
            std::vector<std::vector<double>> out;
            if (!arr.is_array())
                return out;
            for (std::size_t i = 0; i < arr.size(); ++i)
            {
                const Json &zone = arr[i];
                if (zone.is_array() && zone.size() >= 3)
                {
                    out.push_back({zone[std::size_t(0)].as_number(),
                                   zone[std::size_t(1)].as_number(),
                                   zone[std::size_t(2)].as_number()});
                }
            }
            return out;
        }

        std::vector<std::string> parse_string_array(const Json &arr)
        {
            std::vector<std::string> out;
            if (!arr.is_array())
                return out;
            for (std::size_t i = 0; i < arr.size(); ++i)
                out.push_back(arr[i].as_string());
            return out;
        }

        std::vector<int> parse_int_array(const Json &arr, const std::vector<int> &def)
        {
            if (!arr.is_array())
                return def;
            std::vector<int> out;
            for (std::size_t i = 0; i < arr.size(); ++i)
                out.push_back(arr[i].as_int());
            return out.empty() ? def : out;
        }

        Json to_json_array(const std::vector<std::string> &v)
        {
            Json a = Json::array();
            for (const auto &s : v)
                a.push_back(Json(s));
            return a;
        }

        Json to_json_array(const std::vector<int> &v)
        {
            Json a = Json::array();
            for (int n : v)
                a.push_back(Json(n));
            return a;
        }

        Json to_json_zones(const std::vector<std::vector<double>> &zones)
        {
            Json a = Json::array();
            for (const auto &z : zones)
            {
                Json inner = Json::array();
                for (double d : z)
                    inner.push_back(Json(d));
                a.push_back(inner);
            }
            return a;
        }

    } // namespace

    bool Config::load(const std::string &path, std::string &error)
    {
        Json root;
        if (!Json::load_file(path, root, error))
            return false;
        if (!root.is_object())
        {
            error = "configuration root must be a JSON object";
            return false;
        }

        // ---- robot ----------------------------------------------------------
        robot.name = root.get_string("robot.name", robot.name);
        robot.model = root.get_string("robot.model", robot.model);
        robot.serial = root.get_string("robot.serial", robot.serial);
        robot.personality_preset =
            root.get_string("robot.personality_preset", robot.personality_preset);

        // ---- brain ----------------------------------------------------------
        brain.learning_rate = root.get_number("brain.learning_rate", brain.learning_rate);
        brain.hidden_layers = parse_int_array(root["brain"]["hidden_layers"],
                                              brain.hidden_layers);
        brain.memory_capacity = root.get_int("brain.memory_capacity", brain.memory_capacity);
        brain.decision_threshold =
            root.get_number("brain.decision_threshold", brain.decision_threshold);
        brain.curiosity = root.get_number("brain.curiosity", brain.curiosity);
        brain.reflection_interval_sec =
            root.get_int("brain.reflection_interval_sec", brain.reflection_interval_sec);

        // ---- personality ----------------------------------------------------
        personality.preset = root.get_string("personality.preset", personality.preset);
        personality.openness = root.get_number("personality.traits.openness", personality.openness);
        personality.conscientiousness =
            root.get_number("personality.traits.conscientiousness", personality.conscientiousness);
        personality.extraversion =
            root.get_number("personality.traits.extraversion", personality.extraversion);
        personality.agreeableness =
            root.get_number("personality.traits.agreeableness", personality.agreeableness);
        personality.neuroticism =
            root.get_number("personality.traits.neuroticism", personality.neuroticism);
        personality.curiosity = root.get_number("personality.traits.curiosity", personality.curiosity);
        personality.playfulness =
            root.get_number("personality.traits.playfulness", personality.playfulness);
        personality.assertiveness =
            root.get_number("personality.traits.assertiveness", personality.assertiveness);
        personality.mood_baseline =
            root.get_number("personality.mood.baseline", personality.mood_baseline);
        personality.mood_volatility =
            root.get_number("personality.mood.volatility", personality.mood_volatility);

        // ---- vision ---------------------------------------------------------
        vision.enabled = root.get_bool("vision.enabled", vision.enabled);
        vision.width = root.get_int("vision.width", vision.width);
        vision.height = root.get_int("vision.height", vision.height);
        vision.channels = root.get_int("vision.channels", vision.channels);
        vision.detection_threshold =
            root.get_number("vision.detection_threshold", vision.detection_threshold);
        vision.frame_history = root.get_int("vision.frame_history", vision.frame_history);

        // ---- speech ---------------------------------------------------------
        speech.recognition_enabled =
            root.get_bool("speech.recognition_enabled", speech.recognition_enabled);
        speech.tts_enabled = root.get_bool("speech.tts_enabled", speech.tts_enabled);
        speech.language = root.get_string("speech.language", speech.language);
        speech.voice = root.get_string("speech.voice", speech.voice);
        speech.wake_word = root.get_string("speech.wake_word", speech.wake_word);
        speech.confidence_threshold =
            root.get_number("speech.confidence_threshold", speech.confidence_threshold);

        // ---- robotics -------------------------------------------------------
        robotics.max_speed = root.get_number("robotics.max_speed", robotics.max_speed);
        robotics.battery_capacity =
            root.get_number("robotics.battery_capacity", robotics.battery_capacity);
        robotics.idle_drain_per_sec =
            root.get_number("robotics.idle_drain_per_sec", robotics.idle_drain_per_sec);
        robotics.move_drain_per_sec =
            root.get_number("robotics.move_drain_per_sec", robotics.move_drain_per_sec);
        robotics.servo_count = root.get_int("robotics.servo_count", robotics.servo_count);
        robotics.arm_reach_cm =
            root.get_number("robotics.arm_reach_cm", robotics.arm_reach_cm);

        // ---- navigation -----------------------------------------------------
        navigation.world_width = root.get_int("navigation.world_width", navigation.world_width);
        navigation.world_height = root.get_int("navigation.world_height", navigation.world_height);
        navigation.obstacle_density =
            root.get_number("navigation.obstacle_density", navigation.obstacle_density);
        navigation.algorithm = root.get_string("navigation.algorithm", navigation.algorithm);
        navigation.step_cost = root.get_number("navigation.step_cost", navigation.step_cost);
        navigation.diagonal = root.get_bool("navigation.diagonal", navigation.diagonal);

        // ---- sensors --------------------------------------------------------
        sensors.range_cm = root.get_number("sensors.range_cm", sensors.range_cm);
        sensors.beam_count = root.get_int("sensors.beam_count", sensors.beam_count);
        sensors.imu_noise = root.get_number("sensors.imu_noise", sensors.imu_noise);
        sensors.temperature_c = root.get_number("sensors.temperature_c", sensors.temperature_c);

        // ---- security -------------------------------------------------------
        security.autonomy_level =
            root.get_string("security.autonomy_level", security.autonomy_level);
        security.require_confirmation =
            parse_string_array(root["security"]["require_confirmation"]);
        if (security.require_confirmation.empty())
            security.require_confirmation = {"shutdown", "reformat", "network_broadcast"};
        security.denied_zones = parse_denied_zones(root["security"]["denied_zones"]);
        security.max_speed_without_permission =
            root.get_number("security.max_speed_without_permission",
                            security.max_speed_without_permission);
        security.audit_log = root.get_bool("security.audit_log", security.audit_log);

        // ---- networking -----------------------------------------------------
        networking.port = root.get_int("networking.port", networking.port);
        networking.bind = root.get_string("networking.bind", networking.bind);
        networking.enable_remote_control =
            root.get_bool("networking.enable_remote_control",
                          networking.enable_remote_control);
        networking.heartbeat_sec =
            root.get_int("networking.heartbeat_sec", networking.heartbeat_sec);

        // ---- plugins --------------------------------------------------------
        plugins.directory = root.get_string("plugins.directory", plugins.directory);
        plugins.auto_load = root.get_bool("plugins.auto_load", plugins.auto_load);

        // ---- database -------------------------------------------------------
        database.path = root.get_string("database.path", database.path);
        database.autosave_sec = root.get_int("database.autosave_sec", database.autosave_sec);

        // ---- logging --------------------------------------------------------
        logging.level = root.get_string("logging.level", logging.level);
        logging.console = root.get_bool("logging.console", logging.console);
        logging.file = root.get_string("logging.file", logging.file);

        source_path_ = path;
        return true;
    }

    Json Config::to_json() const
    {
        Json root = Json::object();

        Json r = Json::object();
        r.set("name", Json(robot.name));
        r.set("model", Json(robot.model));
        r.set("serial", Json(robot.serial));
        r.set("personality_preset", Json(robot.personality_preset));
        root.set("robot", r);

        Json b = Json::object();
        b.set("learning_rate", Json(brain.learning_rate));
        b.set("hidden_layers", to_json_array(brain.hidden_layers));
        b.set("memory_capacity", Json(brain.memory_capacity));
        b.set("decision_threshold", Json(brain.decision_threshold));
        b.set("curiosity", Json(brain.curiosity));
        b.set("reflection_interval_sec", Json(brain.reflection_interval_sec));
        root.set("brain", b);

        Json p = Json::object();
        p.set("preset", Json(personality.preset));
        Json traits = Json::object();
        traits.set("openness", Json(personality.openness));
        traits.set("conscientiousness", Json(personality.conscientiousness));
        traits.set("extraversion", Json(personality.extraversion));
        traits.set("agreeableness", Json(personality.agreeableness));
        traits.set("neuroticism", Json(personality.neuroticism));
        traits.set("curiosity", Json(personality.curiosity));
        traits.set("playfulness", Json(personality.playfulness));
        traits.set("assertiveness", Json(personality.assertiveness));
        p.set("traits", traits);
        Json mood = Json::object();
        mood.set("baseline", Json(personality.mood_baseline));
        mood.set("volatility", Json(personality.mood_volatility));
        p.set("mood", mood);
        root.set("personality", p);

        Json v = Json::object();
        v.set("enabled", Json(vision.enabled));
        v.set("width", Json(vision.width));
        v.set("height", Json(vision.height));
        v.set("channels", Json(vision.channels));
        v.set("detection_threshold", Json(vision.detection_threshold));
        v.set("frame_history", Json(vision.frame_history));
        root.set("vision", v);

        Json s = Json::object();
        s.set("recognition_enabled", Json(speech.recognition_enabled));
        s.set("tts_enabled", Json(speech.tts_enabled));
        s.set("language", Json(speech.language));
        s.set("voice", Json(speech.voice));
        s.set("wake_word", Json(speech.wake_word));
        s.set("confidence_threshold", Json(speech.confidence_threshold));
        root.set("speech", s);

        Json rb = Json::object();
        rb.set("max_speed", Json(robotics.max_speed));
        rb.set("battery_capacity", Json(robotics.battery_capacity));
        rb.set("idle_drain_per_sec", Json(robotics.idle_drain_per_sec));
        rb.set("move_drain_per_sec", Json(robotics.move_drain_per_sec));
        rb.set("servo_count", Json(robotics.servo_count));
        rb.set("arm_reach_cm", Json(robotics.arm_reach_cm));
        root.set("robotics", rb);

        Json n = Json::object();
        n.set("world_width", Json(navigation.world_width));
        n.set("world_height", Json(navigation.world_height));
        n.set("obstacle_density", Json(navigation.obstacle_density));
        n.set("algorithm", Json(navigation.algorithm));
        n.set("step_cost", Json(navigation.step_cost));
        n.set("diagonal", Json(navigation.diagonal));
        root.set("navigation", n);

        Json sn = Json::object();
        sn.set("range_cm", Json(sensors.range_cm));
        sn.set("beam_count", Json(sensors.beam_count));
        sn.set("imu_noise", Json(sensors.imu_noise));
        sn.set("temperature_c", Json(sensors.temperature_c));
        root.set("sensors", sn);

        Json sec = Json::object();
        sec.set("autonomy_level", Json(security.autonomy_level));
        sec.set("require_confirmation", to_json_array(security.require_confirmation));
        sec.set("denied_zones", to_json_zones(security.denied_zones));
        sec.set("max_speed_without_permission", Json(security.max_speed_without_permission));
        sec.set("audit_log", Json(security.audit_log));
        root.set("security", sec);

        Json net = Json::object();
        net.set("port", Json(networking.port));
        net.set("bind", Json(networking.bind));
        net.set("enable_remote_control", Json(networking.enable_remote_control));
        net.set("heartbeat_sec", Json(networking.heartbeat_sec));
        root.set("networking", net);

        Json pl = Json::object();
        pl.set("directory", Json(plugins.directory));
        pl.set("auto_load", Json(plugins.auto_load));
        root.set("plugins", pl);

        Json db = Json::object();
        db.set("path", Json(database.path));
        db.set("autosave_sec", Json(database.autosave_sec));
        root.set("database", db);

        Json lg = Json::object();
        lg.set("level", Json(logging.level));
        lg.set("console", Json(logging.console));
        lg.set("file", Json(logging.file));
        root.set("logging", lg);

        return root;
    }

    bool Config::save(const std::string &path) const
    {
        return to_json().save_file(path);
    }

} // namespace chappie
