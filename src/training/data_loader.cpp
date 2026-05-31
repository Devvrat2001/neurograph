#include "neurograph/training/data_loader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>
#include <cmath>
#include <cctype>
#include <filesystem>

namespace neurograph::training {

static std::vector<std::string> s_last_class_labels;

static std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;

    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            // Trim whitespace
            size_t start = field.find_first_not_of(" \t");
            size_t end = field.find_last_not_of(" \t\r\n");
            if (start != std::string::npos)
                fields.push_back(field.substr(start, end - start + 1));
            else
                fields.push_back("");
            field.clear();
        } else if (c != '\r') {
            field += c;
        }
    }
    // Last field
    size_t start = field.find_first_not_of(" \t");
    size_t end = field.find_last_not_of(" \t\r\n");
    if (start != std::string::npos)
        fields.push_back(field.substr(start, end - start + 1));
    else
        fields.push_back("");

    return fields;
}

static bool is_numeric(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

util::Result<std::vector<Sample>> DataLoader::load_csv(
    const std::string& path, int num_target_cols,
    bool has_header, bool auto_one_hot)
{
    std::ifstream file(path);
    if (!file.is_open())
        return util::Error("Cannot open file: " + path);

    std::string line;
    std::vector<std::vector<std::string>> rows;

    // Skip header
    if (has_header && std::getline(file, line)) {
        // Header discarded
    }

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto fields = split_csv_line(line);
        if (!fields.empty())
            rows.push_back(std::move(fields));
    }

    if (rows.empty())
        return util::Error("CSV file is empty or has no data rows");

    int total_cols = static_cast<int>(rows[0].size());
    int feature_cols = total_cols - num_target_cols;
    if (feature_cols <= 0)
        return util::Error("Not enough columns for features + targets");

    // Detect if target columns are non-numeric (classification with string labels)
    bool target_is_string = false;
    if (auto_one_hot && num_target_cols == 1) {
        for (auto& row : rows) {
            if (static_cast<int>(row.size()) > feature_cols &&
                !is_numeric(row[feature_cols])) {
                target_is_string = true;
                break;
            }
        }
    }

    // Build class label mapping for one-hot encoding
    std::unordered_map<std::string, int> label_map;
    std::vector<std::string> class_labels;

    if (target_is_string) {
        for (auto& row : rows) {
            if (static_cast<int>(row.size()) <= feature_cols) continue;
            const auto& label = row[feature_cols];
            if (label_map.find(label) == label_map.end()) {
                label_map[label] = static_cast<int>(class_labels.size());
                class_labels.push_back(label);
            }
        }
    }

    // Build samples
    std::vector<Sample> samples;
    samples.reserve(rows.size());

    for (auto& row : rows) {
        if (static_cast<int>(row.size()) < total_cols) continue;

        Sample sample;
        sample.features.resize(feature_cols);
        for (int i = 0; i < feature_cols; ++i) {
            sample.features[i] = is_numeric(row[i]) ? std::stod(row[i]) : 0.0;
        }

        if (target_is_string) {
            // One-hot encode
            int num_classes = static_cast<int>(class_labels.size());
            sample.target.resize(num_classes, 0.0);
            auto it = label_map.find(row[feature_cols]);
            if (it != label_map.end())
                sample.target[it->second] = 1.0;
            sample.label = row[feature_cols];
        } else {
            // Numeric targets
            sample.target.resize(num_target_cols);
            for (int i = 0; i < num_target_cols; ++i)
                sample.target[i] = is_numeric(row[feature_cols + i])
                    ? std::stod(row[feature_cols + i]) : 0.0;
        }

        samples.push_back(std::move(sample));
    }

    s_last_class_labels = class_labels;
    return samples;
}

DataSplit DataLoader::train_test_split(
    const std::vector<Sample>& data, double test_ratio,
    bool shuffle, uint64_t seed)
{
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::mt19937_64 rng(seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    size_t test_count = static_cast<size_t>(data.size() * test_ratio);
    size_t train_count = data.size() - test_count;

    DataSplit split;
    split.train.reserve(train_count);
    split.test.reserve(test_count);

    for (size_t i = 0; i < train_count; ++i)
        split.train.push_back(data[indices[i]]);
    for (size_t i = train_count; i < data.size(); ++i)
        split.test.push_back(data[indices[i]]);

    return split;
}

std::vector<DataBatch> DataLoader::make_batches(
    const std::vector<Sample>& data, size_t batch_size)
{
    std::vector<DataBatch> batches;
    for (size_t i = 0; i < data.size(); i += batch_size) {
        DataBatch batch;
        batch.batch_index = i / batch_size;
        size_t end = std::min(i + batch_size, data.size());
        batch.samples.assign(data.begin() + i, data.begin() + end);
        batches.push_back(std::move(batch));
    }
    return batches;
}

std::vector<NormStats> DataLoader::compute_stats(const std::vector<Sample>& data) {
    if (data.empty()) return {};
    size_t n_features = data[0].features.size();
    std::vector<NormStats> stats(n_features);

    for (size_t f = 0; f < n_features; ++f) {
        double sum = 0.0, min_v = 1e30, max_v = -1e30;
        for (auto& s : data) {
            sum += s.features[f];
            min_v = std::min(min_v, s.features[f]);
            max_v = std::max(max_v, s.features[f]);
        }
        stats[f].mean = sum / data.size();
        stats[f].min_val = min_v;
        stats[f].max_val = max_v;

        double var_sum = 0.0;
        for (auto& s : data)
            var_sum += (s.features[f] - stats[f].mean) * (s.features[f] - stats[f].mean);
        stats[f].stddev = std::sqrt(var_sum / data.size());
        if (stats[f].stddev < 1e-12) stats[f].stddev = 1.0;
    }
    return stats;
}

void DataLoader::normalize_zscore(std::vector<Sample>& data,
                                   const std::vector<NormStats>& stats) {
    for (auto& s : data)
        for (size_t f = 0; f < s.features.size() && f < stats.size(); ++f)
            s.features[f] = (s.features[f] - stats[f].mean) / stats[f].stddev;
}

void DataLoader::normalize_minmax(std::vector<Sample>& data,
                                   const std::vector<NormStats>& stats) {
    for (auto& s : data) {
        for (size_t f = 0; f < s.features.size() && f < stats.size(); ++f) {
            double range = stats[f].max_val - stats[f].min_val;
            if (range < 1e-12) range = 1.0;
            s.features[f] = (s.features[f] - stats[f].min_val) / range;
        }
    }
}

// MNIST IDX format helpers
static uint32_t read_u32_be(std::ifstream& f) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) |
           static_cast<uint32_t>(bytes[3]);
}

util::Result<std::vector<Sample>> DataLoader::load_mnist_idx(
    const std::string& images_path,
    const std::string& labels_path,
    int max_samples)
{
    std::ifstream img_file(images_path, std::ios::binary);
    if (!img_file.is_open())
        return util::Error("Cannot open MNIST images file: " + images_path);

    std::ifstream lbl_file(labels_path, std::ios::binary);
    if (!lbl_file.is_open())
        return util::Error("Cannot open MNIST labels file: " + labels_path);

    // Read image file header
    uint32_t img_magic = read_u32_be(img_file);
    if (img_magic != 0x00000803)
        return util::Error("Invalid MNIST image magic number: " + std::to_string(img_magic));

    uint32_t num_images = read_u32_be(img_file);
    uint32_t rows = read_u32_be(img_file);
    uint32_t cols = read_u32_be(img_file);

    if (rows != 28 || cols != 28)
        return util::Error("Unexpected MNIST image size: " +
            std::to_string(rows) + "x" + std::to_string(cols));

    // Read label file header
    uint32_t lbl_magic = read_u32_be(lbl_file);
    if (lbl_magic != 0x00000801)
        return util::Error("Invalid MNIST label magic number");

    uint32_t num_labels = read_u32_be(lbl_file);
    if (num_labels != num_images)
        return util::Error("MNIST image/label count mismatch");

    int count = static_cast<int>(num_images);
    if (max_samples > 0 && max_samples < count)
        count = max_samples;

    std::vector<Sample> samples;
    samples.reserve(count);

    int pixel_count = static_cast<int>(rows * cols); // 784

    for (int i = 0; i < count; ++i) {
        Sample sample;
        sample.features.resize(pixel_count);

        // Read pixel data (unsigned bytes, 0-255)
        std::vector<uint8_t> pixels(pixel_count);
        img_file.read(reinterpret_cast<char*>(pixels.data()), pixel_count);
        if (!img_file)
            return util::Error("Unexpected end of MNIST image file at sample " + std::to_string(i));

        for (int p = 0; p < pixel_count; ++p)
            sample.features[p] = static_cast<double>(pixels[p]) / 255.0;

        // Read label (0-9)
        uint8_t label;
        lbl_file.read(reinterpret_cast<char*>(&label), 1);
        if (!lbl_file)
            return util::Error("Unexpected end of MNIST label file at sample " + std::to_string(i));

        // One-hot encode (10 classes)
        sample.target.resize(10, 0.0);
        sample.target[label] = 1.0;
        sample.label = std::to_string(static_cast<int>(label));

        samples.push_back(std::move(sample));
    }

    // Set class labels
    s_last_class_labels = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

    return samples;
}

const std::vector<std::string>& DataLoader::last_class_labels() {
    return s_last_class_labels;
}

void DataLoader::set_class_labels(const std::vector<std::string>& labels) {
    s_last_class_labels = labels;
}

// --- BMP / image folder support ---

std::vector<uint8_t> DataLoader::read_bmp_pixels(
    const std::string& path, int& width, int& height, int& channels)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return {};

    // Read the 54-byte BMP header (minimum)
    uint8_t header[54];
    file.read(reinterpret_cast<char*>(header), 54);
    if (!file || header[0] != 'B' || header[1] != 'M')
        return {};

    // Pixel data offset (bytes 10-13, little-endian)
    uint32_t data_offset = static_cast<uint32_t>(header[10])
        | (static_cast<uint32_t>(header[11]) << 8)
        | (static_cast<uint32_t>(header[12]) << 16)
        | (static_cast<uint32_t>(header[13]) << 24);

    // Image width (bytes 18-21, little-endian, signed)
    int32_t w = static_cast<int32_t>(
        static_cast<uint32_t>(header[18])
        | (static_cast<uint32_t>(header[19]) << 8)
        | (static_cast<uint32_t>(header[20]) << 16)
        | (static_cast<uint32_t>(header[21]) << 24));

    // Image height (bytes 22-25, little-endian, signed; negative = top-down)
    int32_t h = static_cast<int32_t>(
        static_cast<uint32_t>(header[22])
        | (static_cast<uint32_t>(header[23]) << 8)
        | (static_cast<uint32_t>(header[24]) << 16)
        | (static_cast<uint32_t>(header[25]) << 24));

    bool top_down = (h < 0);
    if (top_down) h = -h;

    // Bits per pixel (bytes 28-29)
    uint16_t bpp = static_cast<uint16_t>(header[28])
        | (static_cast<uint16_t>(header[29]) << 8);

    if (bpp != 24 && bpp != 32)
        return {};

    int bytes_per_pixel = bpp / 8;
    channels = 3; // We always output RGB
    width = w;
    height = h;

    // BMP rows are padded to 4-byte boundaries
    int row_size = ((w * bytes_per_pixel + 3) / 4) * 4;

    // Seek to pixel data
    file.seekg(data_offset, std::ios::beg);

    std::vector<uint8_t> raw_row(row_size);
    std::vector<uint8_t> pixels(w * h * 3);

    for (int y = 0; y < h; ++y) {
        file.read(reinterpret_cast<char*>(raw_row.data()), row_size);
        if (!file) return {};

        // BMP stores rows bottom-to-top (unless top_down)
        int dest_y = top_down ? y : (h - 1 - y);

        for (int x = 0; x < w; ++x) {
            int src_idx = x * bytes_per_pixel;
            int dst_idx = (dest_y * w + x) * 3;
            // BMP stores BGR; convert to RGB
            pixels[dst_idx + 0] = raw_row[src_idx + 2]; // R
            pixels[dst_idx + 1] = raw_row[src_idx + 1]; // G
            pixels[dst_idx + 2] = raw_row[src_idx + 0]; // B
        }
    }

    return pixels;
}

std::vector<double> DataLoader::resize_image(
    const std::vector<double>& pixels,
    int src_w, int src_h,
    int dst_w, int dst_h)
{
    std::vector<double> result(dst_w * dst_h);
    for (int y = 0; y < dst_h; ++y) {
        int src_y = y * src_h / dst_h;
        if (src_y >= src_h) src_y = src_h - 1;
        for (int x = 0; x < dst_w; ++x) {
            int src_x = x * src_w / dst_w;
            if (src_x >= src_w) src_x = src_w - 1;
            result[y * dst_w + x] = pixels[src_y * src_w + src_x];
        }
    }
    return result;
}

util::Result<std::vector<Sample>> DataLoader::load_image_folder(
    const std::string& root_path,
    int target_width,
    int target_height,
    int max_samples)
{
    namespace fs = std::filesystem;

    if (!fs::exists(root_path) || !fs::is_directory(root_path))
        return util::Error("Image folder does not exist or is not a directory: " + root_path);

    // Collect subdirectories (each = one class), sorted alphabetically
    std::vector<std::string> class_dirs;
    for (auto& entry : fs::directory_iterator(root_path)) {
        if (entry.is_directory()) {
            class_dirs.push_back(entry.path().filename().string());
        }
    }
    std::sort(class_dirs.begin(), class_dirs.end());

    if (class_dirs.empty())
        return util::Error("No class subdirectories found in: " + root_path);

    int num_classes = static_cast<int>(class_dirs.size());

    // Build samples from each class directory
    std::vector<Sample> samples;

    for (int cls = 0; cls < num_classes; ++cls) {
        fs::path class_path = fs::path(root_path) / class_dirs[cls];

        for (auto& entry : fs::directory_iterator(class_path)) {
            if (!entry.is_regular_file()) continue;

            // Check for .bmp extension (case-insensitive)
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext != ".bmp") continue;

            int img_w = 0, img_h = 0, img_channels = 0;
            auto pixels = read_bmp_pixels(
                entry.path().string(), img_w, img_h, img_channels);
            if (pixels.empty() || img_w <= 0 || img_h <= 0)
                continue; // Skip unreadable images

            // Convert to grayscale and normalize to [0, 1]
            std::vector<double> gray(img_w * img_h);
            for (int i = 0; i < img_w * img_h; ++i) {
                double r = static_cast<double>(pixels[i * 3 + 0]);
                double g = static_cast<double>(pixels[i * 3 + 1]);
                double b = static_cast<double>(pixels[i * 3 + 2]);
                gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
            }

            // Resize to target dimensions
            std::vector<double> resized = resize_image(
                gray, img_w, img_h, target_width, target_height);

            // Build sample
            Sample sample;
            sample.features = std::move(resized);
            sample.target.resize(num_classes, 0.0);
            sample.target[cls] = 1.0;
            sample.label = class_dirs[cls];

            samples.push_back(std::move(sample));
        }
    }

    if (samples.empty())
        return util::Error("No valid .bmp images found in: " + root_path);

    // Shuffle samples (deterministic seed)
    {
        std::mt19937_64 rng(42);
        std::shuffle(samples.begin(), samples.end(), rng);
    }

    // Limit to max_samples if specified
    if (max_samples > 0 && static_cast<int>(samples.size()) > max_samples) {
        samples.resize(max_samples);
    }

    // Store class labels
    s_last_class_labels = class_dirs;

    return samples;
}

} // namespace neurograph::training
