#pragma once

#include "neurograph/training/training_engine.hpp"

#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace neurograph::training {

class MetricsExporter {
public:
    /// Export training history to CSV file
    static void export_csv(const std::string& path,
                           const std::vector<EpochResult>& history)
    {
        // Ensure parent directory exists
        std::filesystem::path p(path);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());

        std::ofstream out(path);
        if (!out.is_open()) return;

        // Header
        out << "epoch,train_loss,train_accuracy,val_loss,val_accuracy,epoch_duration_ms\n";

        // Data rows
        out << std::fixed;
        for (const auto& r : history) {
            out << r.epoch << ","
                << std::setprecision(6) << r.train_loss << ","
                << std::setprecision(6) << r.train_accuracy << ","
                << std::setprecision(6) << r.val_loss << ","
                << std::setprecision(6) << r.val_accuracy << ","
                << std::setprecision(1) << r.epoch_duration_ms << "\n";
        }
    }

    /// Export as JSON for programmatic use
    static void export_json(const std::string& path,
                            const std::vector<EpochResult>& history,
                            const std::string& model_name = "",
                            const std::string& dataset_name = "")
    {
        // Ensure parent directory exists
        std::filesystem::path p(path);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());

        std::ofstream out(path);
        if (!out.is_open()) return;

        // Compute summary stats
        double best_val_accuracy = 0.0;
        double best_val_loss = 1e30;
        for (const auto& r : history) {
            if (r.val_accuracy > best_val_accuracy) best_val_accuracy = r.val_accuracy;
            if (r.val_loss < best_val_loss) best_val_loss = r.val_loss;
        }

        // Timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf{};
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t_now);
#else
        localtime_r(&time_t_now, &tm_buf);
#endif
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", &tm_buf);

        out << std::fixed;
        out << "{\n";
        out << "  \"model_name\": \"" << escape_json(model_name) << "\",\n";
        out << "  \"dataset_name\": \"" << escape_json(dataset_name) << "\",\n";
        out << "  \"timestamp\": \"" << ts << "\",\n";
        out << "  \"total_epochs\": " << static_cast<int>(history.size()) << ",\n";
        out << std::setprecision(6);
        out << "  \"best_val_accuracy\": " << best_val_accuracy << ",\n";
        out << "  \"best_val_loss\": " << best_val_loss << ",\n";
        out << "  \"history\": [\n";

        for (size_t i = 0; i < history.size(); ++i) {
            const auto& r = history[i];
            out << "    {";
            out << "\"epoch\": " << r.epoch << ", ";
            out << "\"train_loss\": " << r.train_loss << ", ";
            out << "\"train_accuracy\": " << r.train_accuracy << ", ";
            out << "\"val_loss\": " << r.val_loss << ", ";
            out << "\"val_accuracy\": " << r.val_accuracy << ", ";
            out << std::setprecision(1);
            out << "\"epoch_duration_ms\": " << r.epoch_duration_ms;
            out << std::setprecision(6);
            out << "}";
            if (i + 1 < history.size()) out << ",";
            out << "\n";
        }

        out << "  ]\n";
        out << "}\n";
    }

    /// Generate a timestamped filename (without extension)
    static std::string timestamped_name()
    {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf{};
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t_now);
#else
        localtime_r(&time_t_now, &tm_buf);
#endif
        char buf[64];
        std::strftime(buf, sizeof(buf), "metrics_%Y%m%d_%H%M%S", &tm_buf);
        return std::string(buf);
    }

private:
    static std::string escape_json(const std::string& s)
    {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '\"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:   out += c;      break;
            }
        }
        return out;
    }
};

} // namespace neurograph::training
