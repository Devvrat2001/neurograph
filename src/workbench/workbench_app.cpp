/// NeuroGraph Workbench — Native ImGui Application
///
/// A desktop GUI for creating, training, tuning, saving, and loading
/// neural networks using the NeuroGraph engine. Runs in the same process
/// as the network — no serialization, no WebSocket, no payload limits.

#include "neurograph/core/network.hpp"
#include "neurograph/core/topology.hpp"
#include "neurograph/snapshot/model_snapshot.hpp"
#include "neurograph/snapshot/snapshot_registry.hpp"
#include "neurograph/training/training_engine.hpp"
#include "neurograph/training/metrics_export.hpp"
#include "neurograph/training/auto_config.hpp"
#include "neurograph/util/logger.hpp"

// P1-09 wizard. The state lives in neurograph_lib (testable); the
// render layer (`wizard_ui.hpp`) lives next to this translation unit
// in src/workbench/wizard/. CMake adds that directory to the include
// path for this target, so the unqualified include works.
#include "neurograph/workbench/wizard_state.hpp"
#include "wizard/wizard_ui.hpp"

// P1-11 — License gate. Headers + state types live here; the actual
// crypto/persistence/trial logic stays in neurograph_lib so the unit
// tests can exercise it without spinning up D3D11.
#include "license_dialog.hpp"

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>

#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <cfloat>
#include <fstream>
#include <unordered_set>

// Windows DPI
#include <shellscalingapi.h>
#pragma comment(lib, "Shcore.lib")

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// =================================================================
// DirectX 11 Globals
// =================================================================
static ID3D11Device*            g_pd3dDevice = nullptr;
static ID3D11DeviceContext*     g_pd3dDeviceContext = nullptr;
static IDXGISwapChain*          g_pSwapChain = nullptr;
static ID3D11RenderTargetView*  g_mainRenderTargetView = nullptr;
static UINT                     g_ResizeWidth = 0, g_ResizeHeight = 0;

static bool CreateDeviceD3D(HWND hWnd);
static void CleanupDeviceD3D();
static void CreateRenderTarget();
static void CleanupRenderTarget();
static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// =================================================================
// DPI Scaling
// =================================================================
static float g_dpiScale = 1.0f;

static float Scaled(float v) { return v * g_dpiScale; }
static ImVec2 Scaled(float x, float y) { return ImVec2(x * g_dpiScale, y * g_dpiScale); }

// =================================================================
// Application State
// =================================================================
using namespace neurograph;

struct AppState {
    // Network
    core::Network network;
    snapshot::SnapshotRegistry registry;
    std::string loaded_image_name;
    std::string models_dir = "./models";

    // Training
    std::atomic<bool> training{false};
    std::thread train_thread;
    int train_steps_done = 0;
    float learning_rate = 0.01f;
    float affinity_threshold = 0.25f;
    float pruning_threshold = 0.1f;
    int stream_interval_ms = 100;

    // Create network dialog
    bool show_create_dialog = false;
    char new_name[256] = "my_network";
    int new_neurons = 50;
    int new_connections = 100;
    float new_lr = 0.01f;
    int new_task_type = 5; // CUSTOM
    char new_desc[512] = "";

    // Save dialog
    char save_name[256] = "";
    char save_desc[512] = "";
    char save_tags[256] = "";
    int save_task_type = 5;

    // Delete dialog
    bool show_delete_dialog = false;
    std::string delete_target;

    // 3D view
    float rot_x = 0.4f, rot_y = 0.6f, zoom = 1.0f;
    bool dragging = false;
    ImVec2 drag_start;

    // Inspector
    std::string selected_neuron_id;

    // Event log
    struct Event {
        std::string time;
        std::string type;
        std::string message;
    };
    std::vector<Event> events;

    // Stats history for charts
    std::vector<float> activation_history;
    std::vector<float> weight_history;
    std::vector<float> neuron_count_history;

    // First run layout flag
    bool first_frame = true;

    // P1-09: when true, draw_frame() short-circuits to render the
    // guided wizard window instead of the legacy advanced panels.
    // Defaults true so first-run users land on the wizard; the
    // wizard's "Show advanced UI" button flips this off, and the
    // View menu's "Show guided wizard" item flips it back on.
    bool wizard_mode = true;

    // P1-11 — License state. `license_status` is the cached snapshot
    // of "what tier am I on right now" so the menu/status bar can
    // render without hitting disk every frame; it's refreshed by
    // boot_license_state() at startup and reload_license_state()
    // after the dialog mutates anything. `license_dialog` is the
    // per-frame state of the modal -- input buffer, last error, and
    // the modal_block flag that gates the whole UI when there's no
    // valid license/trial.
    workbench::LicenseStatus       license_status;
    workbench::LicenseDialogState  license_dialog;

    // --- Supervised Training ---
    std::unique_ptr<training::TrainingEngine> train_engine;
    char dataset_path[512] = "";
    int num_target_cols = 1;
    bool dataset_has_header = true;
    bool dataset_loaded = false;
    std::string dataset_status;
    // Supervised config
    int sup_epochs = 100;
    int sup_batch_size = 32;
    float sup_lr = 0.001f;
    int sup_optimizer = 1;  // 0=SGD, 1=Adam
    int sup_loss = 1;       // 0=MSE, 1=CrossEntropy
    bool sup_interleave = false;
    int sup_self_org_interval = 10;
    bool sup_use_stall_counter = true;  // gate growth on loss plateau
    int sup_growth_patience = 10;       // cycles of no improvement before grow
    bool auto_configure_on_load = true; // one-click training setup
    // Phase 1 features
    int sup_schedule = 1;    // 0=Constant, 1=Cosine, 2=Exponential
    int sup_warmup = 5;
    float sup_dropout = 0.2f;
    bool sup_batch_norm = true;
    bool sup_augmentation = false;
    std::atomic<bool> sup_training{false};
    std::thread sup_train_thread;
    // Training history for plots (mutex-protected for thread safety)
    std::mutex history_mutex;
    std::vector<float> loss_history;
    std::vector<float> accuracy_history;
    std::vector<float> val_loss_history;
    std::vector<float> val_accuracy_history;

    // --- Inference ---
    char inference_input[1024] = "";
    std::vector<double> last_prediction;
    int last_predicted_class = -1;
    std::string last_predicted_label;

    // --- MNIST ---
    bool is_mnist = false;
    int dataset_type = 0;  // 0=CSV, 1=MNIST, 2=Image Folder
    char image_folder_path[512] = "";
    int image_target_w = 28;
    int image_target_h = 28;
    char mnist_images_path[512] = "";
    char mnist_labels_path[512] = "";
    int mnist_max_samples = 5000;
    std::vector<float> last_mnist_image; // 28x28 pixel values for display

    AppState() : registry("./models") {}
};

static AppState g_app;

// P1-09 — Wizard state lives outside AppState. WizardState holds a
// std::mutex + atomics, so it isn't copyable/movable; keeping it as
// a separate file-static with the same lifetime as g_app sidesteps
// any chance of an accidental copy through AppState's API.
static workbench::WizardState                      g_wizard_state;
static std::unique_ptr<workbench::WizardRuntime>   g_wizard_runtime;

// =================================================================
// Helpers
// =================================================================
static void add_event(const std::string& type, const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&time));
    g_app.events.insert(g_app.events.begin(), {buf, type, msg});
    if (g_app.events.size() > 200) g_app.events.resize(200);
}

static const char* task_type_names[] = {
    "Classification", "Regression", "Clustering", "Association", "Anomaly", "Custom"
};

static snapshot::TaskType task_type_from_idx(int idx) {
    switch (idx) {
        case 0: return snapshot::TaskType::CLASSIFICATION;
        case 1: return snapshot::TaskType::REGRESSION;
        case 2: return snapshot::TaskType::CLUSTERING;
        case 3: return snapshot::TaskType::ASSOCIATION;
        case 4: return snapshot::TaskType::ANOMALY;
        default: return snapshot::TaskType::CUSTOM;
    }
}

static void refresh_registry() {
    g_app.registry.scan();
}

static void create_network() {
    // Clear existing
    {
        std::vector<core::SynapseId> sids;
        for (auto& [id, _] : g_app.network.synapses()) sids.push_back(id);
        for (auto& id : sids) g_app.network.remove_synapse(id);

        std::vector<core::NeuronId> nids;
        for (auto& [id, _] : g_app.network.neurons()) nids.push_back(id);
        for (auto& id : nids) g_app.network.remove_neuron(id);
    }

    std::mt19937 rng(static_cast<unsigned>(
        std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<double> pos(-5.0, 5.0);
    std::uniform_real_distribution<double> wt(0.1, 1.0);

    std::vector<core::NeuronId> ids;
    for (int i = 0; i < g_app.new_neurons; ++i) {
        auto id = core::NeuronId::generate();
        auto neuron = core::Neuron(id, std::string(g_app.new_name) + "_n" + std::to_string(i));
        neuron.set_position(core::Vec3{pos(rng), pos(rng), pos(rng)});
        neuron.set_threshold(0.3 + 0.4 * (rng() % 100) / 100.0);
        g_app.network.add_neuron(std::move(neuron));
        ids.push_back(id);
    }

    if (g_app.new_neurons > 1) {
        std::uniform_int_distribution<int> idx(0, g_app.new_neurons - 1);
        for (int i = 0; i < g_app.new_connections; ++i) {
            int s = idx(rng), t = idx(rng);
            if (s != t) g_app.network.connect(ids[s], ids[t], wt(rng));
        }
    }

    core::TopologyConfig tc;
    tc.learning_rate = g_app.new_lr;
    g_app.network.set_topology_config(tc);

    g_app.loaded_image_name = g_app.new_name;
    add_event("CREATE", std::string("Created: ") + g_app.new_name +
              " (" + std::to_string(g_app.new_neurons) + "n, " +
              std::to_string(g_app.new_connections) + "s)");
}

static void load_image(const std::string& name) {
    auto result = g_app.registry.load(name);
    if (!result.ok()) {
        add_event("ERROR", "Load failed: " + result.error().message);
        return;
    }

    // Clear and copy
    {
        std::vector<core::SynapseId> sids;
        for (auto& [id, _] : g_app.network.synapses()) sids.push_back(id);
        for (auto& id : sids) g_app.network.remove_synapse(id);

        std::vector<core::NeuronId> nids;
        for (auto& [id, _] : g_app.network.neurons()) nids.push_back(id);
        for (auto& id : nids) g_app.network.remove_neuron(id);
    }

    auto& loaded = result.value().network;
    for (auto& [id, neuron] : loaded.neurons())
        g_app.network.add_neuron(core::Neuron(neuron));
    for (auto& [id, synapse] : loaded.synapses())
        g_app.network.add_synapse(core::Synapse(synapse));

    g_app.loaded_image_name = name;
    add_event("LOAD", "Loaded: " + name);
}

static void save_image(const std::string& name, bool overwrite = false) {
    snapshot::SnapshotMetadata meta;
    meta.name = name;
    meta.description = g_app.save_desc;
    meta.task_type = task_type_from_idx(g_app.save_task_type);
    meta.task_name = task_type_names[g_app.save_task_type];
    meta.training.learning_rate = g_app.learning_rate;
    meta.training.self_org_iterations = g_app.network.stats().organization_cycle;
    meta.metrics.avg_activation = g_app.network.stats().avg_activation;
    meta.metrics.avg_weight = g_app.network.stats().avg_weight;

    // Parse tags
    std::string tags_str(g_app.save_tags);
    size_t pos = 0;
    while ((pos = tags_str.find(',')) != std::string::npos) {
        auto tag = tags_str.substr(0, pos);
        tag.erase(0, tag.find_first_not_of(' '));
        tag.erase(tag.find_last_not_of(' ') + 1);
        if (!tag.empty()) meta.tags.push_back(tag);
        tags_str.erase(0, pos + 1);
    }
    tags_str.erase(0, tags_str.find_first_not_of(' '));
    tags_str.erase(tags_str.find_last_not_of(' ') + 1);
    if (!tags_str.empty()) meta.tags.push_back(tags_str);

    if (overwrite && g_app.registry.has(name))
        g_app.registry.remove(name);

    auto result = g_app.registry.save(g_app.network, meta);
    if (result.ok()) {
        add_event("SAVE", "Saved: " + name);
        g_app.loaded_image_name = name;
        refresh_registry();
    } else {
        add_event("ERROR", "Save failed: " + result.error().message);
    }
}

static void delete_image(const std::string& name) {
    auto result = g_app.registry.remove(name);
    if (result.ok()) {
        add_event("DELETE", "Deleted: " + name);
        if (g_app.loaded_image_name == name) g_app.loaded_image_name.clear();
        refresh_registry();
    } else {
        add_event("ERROR", "Delete failed: " + result.error().message);
    }
}

static void start_training() {
    if (g_app.training.load()) return;
    g_app.training = true;
    g_app.train_thread = std::thread([]() {
        while (g_app.training.load()) {
            g_app.network.self_organize(1);
            g_app.train_steps_done++;
            std::this_thread::sleep_for(
                std::chrono::milliseconds(g_app.stream_interval_ms));
        }
    });
    add_event("TRAIN", "Training started");
}

static void stop_training() {
    g_app.training = false;
    if (g_app.train_thread.joinable()) g_app.train_thread.join();
    add_event("TRAIN", "Training stopped at step " + std::to_string(g_app.train_steps_done));
}

// =================================================================
// 3D Projection
// =================================================================
struct Projected {
    float sx, sy, depth, scale;
};

static Projected project(float x, float y, float z, float cx, float cy, float zoom) {
    float cosA = cosf(g_app.rot_y), sinA = sinf(g_app.rot_y);
    float cosB = cosf(g_app.rot_x), sinB = sinf(g_app.rot_x);
    float x1 = x * cosA - z * sinA;
    float z1 = x * sinA + z * cosA;
    float y1 = y * cosB - z1 * sinB;
    float z2 = y * sinB + z1 * cosB;
    float d = 600.0f * zoom;
    float s = d / (d + z2 + 300.0f);
    return { cx + x1 * s * 30.0f * g_dpiScale, cy + y1 * s * 30.0f * g_dpiScale, z2, s };
}

// =================================================================
// Section Header Helper
// =================================================================
static void SectionHeader(const char* label) {
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.45f, 0.72f, 1.0f, 1.0f));
    ImGui::SeparatorText(label);
    ImGui::PopStyleColor();
    ImGui::Spacing();
}

// =================================================================
// UI Drawing
// =================================================================

static void draw_file_panel() {
    // New Network button - prominent accent color
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.47f, 0.87f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.55f, 0.95f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.12f, 0.40f, 0.78f, 1.0f));
    if (ImGui::Button("+ New Network", ImVec2(-1, Scaled(36)))) {
        g_app.show_create_dialog = true;
    }
    ImGui::PopStyleColor(3);

    ImGui::Spacing();
    if (ImGui::Button("Refresh", ImVec2(-1, Scaled(28)))) {
        refresh_registry();
    }

    SectionHeader("SAVED MODELS");

    ImGui::TextColored(ImVec4(0.55f, 0.55f, 0.60f, 1.0f), "%d neural image(s) in ./models",
                       (int)g_app.registry.count());
    ImGui::Spacing();

    ImGui::BeginChild("FileList", ImVec2(0, 0), ImGuiChildFlags_Borders);
    for (auto& [name, entry] : g_app.registry.entries()) {
        bool is_loaded = (name == g_app.loaded_image_name);

        ImGui::PushID(name.c_str());

        // Card-like background for each entry
        ImVec2 cursor = ImGui::GetCursorScreenPos();
        float avail_w = ImGui::GetContentRegionAvail().x;
        ImVec4 card_bg = is_loaded
            ? ImVec4(0.08f, 0.22f, 0.12f, 1.0f)
            : ImVec4(0.08f, 0.10f, 0.15f, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, card_bg);
        ImGui::BeginChild("card", ImVec2(avail_w, Scaled(90)), ImGuiChildFlags_Borders);

        // Name with badge
        if (is_loaded) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.30f, 0.90f, 0.50f, 1.0f));
            ImGui::Text("%s", name.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.20f, 0.70f, 0.35f, 1.0f), " [ACTIVE]");
            ImGui::PopStyleColor();
        } else {
            ImGui::TextColored(ImVec4(0.85f, 0.88f, 0.95f, 1.0f), "%s", name.c_str());
        }

        // Metadata row
        ImGui::TextColored(ImVec4(0.50f, 0.52f, 0.58f, 1.0f), "%s | %d neurons, %d synapses",
            snapshot::task_type_str(entry.metadata.task_type),
            entry.metadata.neuron_count, entry.metadata.synapse_count);

        ImGui::Spacing();

        // Action buttons
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.40f, 0.70f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.50f, 0.85f, 1.0f));
        if (ImGui::Button("Load", Scaled(70, 24))) load_image(name);
        ImGui::PopStyleColor(2);
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.60f, 0.15f, 0.15f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.80f, 0.20f, 0.20f, 1.0f));
        if (ImGui::Button("Delete", Scaled(70, 24))) {
            g_app.delete_target = name;
            g_app.show_delete_dialog = true;
        }
        ImGui::PopStyleColor(2);

        ImGui::EndChild();
        ImGui::PopStyleColor(); // card bg
        ImGui::Spacing();
        ImGui::PopID();
    }
    ImGui::EndChild();
}

static void draw_3d_view(ImVec2 canvas_pos, ImVec2 canvas_size) {
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Background gradient simulation (dark with subtle vignette)
    dl->AddRectFilled(canvas_pos,
        ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
        IM_COL32(8, 11, 20, 255));

    // Grid lines for depth perception
    float cx = canvas_pos.x + canvas_size.x * 0.5f;
    float cy = canvas_pos.y + canvas_size.y * 0.5f;

    // Subtle grid
    for (int i = -4; i <= 4; i++) {
        float offset = i * Scaled(50);
        dl->AddLine(
            ImVec2(canvas_pos.x, cy + offset),
            ImVec2(canvas_pos.x + canvas_size.x, cy + offset),
            IM_COL32(30, 35, 50, 80), 1.0f);
        dl->AddLine(
            ImVec2(cx + offset, canvas_pos.y),
            ImVec2(cx + offset, canvas_pos.y + canvas_size.y),
            IM_COL32(30, 35, 50, 80), 1.0f);
    }

    // Mouse interaction for rotation
    if (ImGui::IsItemHovered()) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            ImVec2 delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
            g_app.rot_y += delta.x * 0.003f;
            g_app.rot_x += delta.y * 0.003f;
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
        }
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0) {
            g_app.zoom = std::clamp(g_app.zoom + wheel * 0.1f, 0.2f, 5.0f);
        }
    }

    auto& neurons = g_app.network.neurons();
    auto& synapses = g_app.network.synapses();

    if (neurons.empty()) {
        // Centered empty state
        const char* msg1 = "No Network Loaded";
        const char* msg2 = "Create a new network or load an existing one";
        ImVec2 ts1 = ImGui::CalcTextSize(msg1);
        ImVec2 ts2 = ImGui::CalcTextSize(msg2);
        dl->AddText(ImVec2(cx - ts1.x * 0.5f, cy - Scaled(16)),
            IM_COL32(80, 90, 110, 200), msg1);
        dl->AddText(ImVec2(cx - ts2.x * 0.5f, cy + Scaled(8)),
            IM_COL32(55, 60, 75, 180), msg2);
        return;
    }

    // Project all neurons
    struct ProjNeuron {
        float sx, sy, depth, scale, activation;
        std::string id;
        bool is_input, is_output;
    };
    std::vector<ProjNeuron> projected;
    projected.reserve(neurons.size());

    std::unordered_map<std::string, size_t> id_to_idx;
    for (auto& [nid, neuron] : neurons) {
        float x = 0, y = 0, z = 0;
        if (neuron.position().has_value()) {
            x = static_cast<float>(neuron.position()->x);
            y = static_cast<float>(neuron.position()->y);
            z = static_cast<float>(neuron.position()->z);
        }
        auto p = project(x, y, z, cx, cy, g_app.zoom);
        bool is_in = neuron.incoming_synapses().empty() && !neuron.outgoing_synapses().empty();
        bool is_out = !neuron.incoming_synapses().empty() && neuron.outgoing_synapses().empty();
        id_to_idx[nid.value()] = projected.size();
        projected.push_back({p.sx, p.sy, p.depth, p.scale,
                            static_cast<float>(neuron.activation_level()),
                            nid.value(), is_in, is_out});
    }

    // Sort back to front
    std::sort(projected.begin(), projected.end(),
        [](const ProjNeuron& a, const ProjNeuron& b) { return a.depth > b.depth; });

    // Draw synapses (limit to 3000 for performance)
    int syn_drawn = 0;
    for (auto& [sid, syn] : synapses) {
        if (syn_drawn >= 3000) break;
        auto it_s = id_to_idx.find(syn.source_id().value());
        auto it_t = id_to_idx.find(syn.target_id().value());
        if (it_s == id_to_idx.end() || it_t == id_to_idx.end()) continue;

        auto& src = projected[it_s->second];
        auto& tgt = projected[it_t->second];
        float w = static_cast<float>(std::abs(syn.weight()));
        int alpha = static_cast<int>(std::min(w * 120.0f, 160.0f));

        ImU32 col = syn.type() == core::SynapseType::EXCITATORY
            ? IM_COL32(70, 140, 255, alpha)
            : IM_COL32(255, 80, 80, alpha);

        dl->AddLine(ImVec2(src.sx, src.sy), ImVec2(tgt.sx, tgt.sy),
                    col, std::max(1.0f, w * 2.0f * g_dpiScale));
        syn_drawn++;
    }

    // Draw neurons
    for (auto& n : projected) {
        float r = std::max(3.0f * g_dpiScale, 6.0f * n.scale * g_dpiScale);
        float a = std::clamp(n.activation * 2.0f + 0.3f, 0.3f, 1.0f);
        int ai = static_cast<int>(a * 255);

        ImU32 col;
        if (n.is_input)       col = IM_COL32(20, 200, 230, ai);   // cyan
        else if (n.is_output) col = IM_COL32(255, 140, 40, ai);   // orange
        else                  col = IM_COL32(50, 210, 110, ai);    // green

        // Glow effect for active neurons
        if (a > 0.6f) {
            dl->AddCircleFilled(ImVec2(n.sx, n.sy), r * 2.0f,
                IM_COL32(ImColor(col).Value.x * 255, ImColor(col).Value.y * 255,
                         ImColor(col).Value.z * 255, 25));
        }

        dl->AddCircleFilled(ImVec2(n.sx, n.sy), r, col);

        // Selection highlight - bright white ring
        if (n.id == g_app.selected_neuron_id) {
            dl->AddCircle(ImVec2(n.sx, n.sy), r + Scaled(4),
                IM_COL32(255, 255, 255, 220), 0, Scaled(2.5f));
            dl->AddCircle(ImVec2(n.sx, n.sy), r + Scaled(7),
                IM_COL32(100, 160, 255, 100), 0, Scaled(1.5f));
        }
    }

    // Click to select
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        ImVec2 mouse = ImGui::GetMousePos();
        float best_dist = Scaled(25);
        std::string best_id;
        for (auto& n : projected) {
            float dx = n.sx - mouse.x, dy = n.sy - mouse.y;
            float d = sqrtf(dx * dx + dy * dy);
            if (d < best_dist) { best_dist = d; best_id = n.id; }
        }
        g_app.selected_neuron_id = best_id;
    }

    // Overlay HUD - top left
    float hud_x = canvas_pos.x + Scaled(14);
    float hud_y = canvas_pos.y + Scaled(10);

    // Semi-transparent background for HUD
    ImVec2 hud_size = Scaled(280, 70);
    dl->AddRectFilled(ImVec2(hud_x - Scaled(4), hud_y - Scaled(4)),
        ImVec2(hud_x + hud_size.x, hud_y + hud_size.y),
        IM_COL32(10, 14, 22, 200), Scaled(6));

    char hud_buf[256];
    snprintf(hud_buf, sizeof(hud_buf), "Neurons: %d", (int)neurons.size());
    dl->AddText(ImVec2(hud_x, hud_y), IM_COL32(120, 180, 255, 230), hud_buf);
    snprintf(hud_buf, sizeof(hud_buf), "Synapses: %d", (int)synapses.size());
    dl->AddText(ImVec2(hud_x, hud_y + Scaled(18)), IM_COL32(120, 180, 255, 230), hud_buf);
    snprintf(hud_buf, sizeof(hud_buf), "Cycle: %d", (int)g_app.network.stats().organization_cycle);
    dl->AddText(ImVec2(hud_x, hud_y + Scaled(36)), IM_COL32(120, 180, 255, 230), hud_buf);

    // Legend - bottom right
    float leg_x = canvas_pos.x + canvas_size.x - Scaled(160);
    float leg_y = canvas_pos.y + canvas_size.y - Scaled(60);
    dl->AddRectFilled(ImVec2(leg_x - Scaled(6), leg_y - Scaled(4)),
        ImVec2(leg_x + Scaled(154), leg_y + Scaled(56)),
        IM_COL32(10, 14, 22, 200), Scaled(6));

    dl->AddCircleFilled(ImVec2(leg_x + Scaled(6), leg_y + Scaled(8)), Scaled(5), IM_COL32(20, 200, 230, 255));
    dl->AddText(ImVec2(leg_x + Scaled(16), leg_y), IM_COL32(160, 170, 185, 220), "Input");

    dl->AddCircleFilled(ImVec2(leg_x + Scaled(6), leg_y + Scaled(26)), Scaled(5), IM_COL32(50, 210, 110, 255));
    dl->AddText(ImVec2(leg_x + Scaled(16), leg_y + Scaled(18)), IM_COL32(160, 170, 185, 220), "Hidden");

    dl->AddCircleFilled(ImVec2(leg_x + Scaled(6), leg_y + Scaled(44)), Scaled(5), IM_COL32(255, 140, 40, 255));
    dl->AddText(ImVec2(leg_x + Scaled(16), leg_y + Scaled(36)), IM_COL32(160, 170, 185, 220), "Output");

    // Controls hint - bottom left
    dl->AddText(ImVec2(canvas_pos.x + Scaled(14), canvas_pos.y + canvas_size.y - Scaled(22)),
        IM_COL32(60, 65, 80, 180), "Drag to rotate | Scroll to zoom | Click to select");
}

static void draw_inspector() {
    if (g_app.selected_neuron_id.empty()) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.45f, 0.48f, 0.55f, 1.0f), "Click a neuron in the 3D view");
        ImGui::TextColored(ImVec4(0.45f, 0.48f, 0.55f, 1.0f), "to inspect its properties.");
        return;
    }

    core::NeuronId nid(g_app.selected_neuron_id);
    auto* neuron = g_app.network.get_neuron(nid);
    if (!neuron) {
        ImGui::Text("Neuron not found.");
        g_app.selected_neuron_id.clear();
        return;
    }

    SectionHeader("NEURON PROPERTIES");

    // ID row
    ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "ID");
    ImGui::SameLine(Scaled(110));
    ImGui::TextColored(ImVec4(0.85f, 0.88f, 0.95f, 1.0f), "%s",
        g_app.selected_neuron_id.substr(0, 12).c_str());

    ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Label");
    ImGui::SameLine(Scaled(110));
    ImGui::TextColored(ImVec4(0.85f, 0.88f, 0.95f, 1.0f), "%s", neuron->label().c_str());

    ImGui::Spacing();

    // Activation bar
    ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Activation");
    float act = static_cast<float>(neuron->activation_level());
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.20f, 0.65f, 0.90f, 1.0f));
    ImGui::ProgressBar(act, ImVec2(-1, Scaled(18)), "");
    ImGui::PopStyleColor();
    ImGui::SameLine(Scaled(4));
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - Scaled(18));
    ImGui::Text("%.6f", act);

    ImGui::Spacing();

    ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Threshold");
    ImGui::SameLine(Scaled(110));
    ImGui::Text("%.4f", neuron->threshold());

    ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "State");
    ImGui::SameLine(Scaled(110));
    auto state_sv = core::neuron_state_str(neuron->state());
    std::string state_str(state_sv);
    ImVec4 state_col = ImVec4(0.85f, 0.88f, 0.95f, 1.0f);
    if (neuron->state() == core::NeuronState::ACTIVE)
        state_col = ImVec4(0.30f, 0.90f, 0.50f, 1.0f);
    else if (neuron->state() == core::NeuronState::DISABLED)
        state_col = ImVec4(0.80f, 0.30f, 0.30f, 1.0f);
    ImGui::TextColored(state_col, "%s", state_str.c_str());

    SectionHeader("CONNECTIONS");

    ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Incoming");
    ImGui::SameLine(Scaled(110));
    ImGui::Text("%d", (int)neuron->incoming_synapses().size());

    ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Outgoing");
    ImGui::SameLine(Scaled(110));
    ImGui::Text("%d", (int)neuron->outgoing_synapses().size());

    // Activation history sparkline
    auto& hist = neuron->activation_history();
    if (!hist.empty()) {
        SectionHeader("HISTORY");
        std::vector<float> fhist;
        size_t start = hist.size() > 50 ? hist.size() - 50 : 0;
        for (size_t i = start; i < hist.size(); ++i)
            fhist.push_back(static_cast<float>(hist[i]));
        ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.30f, 0.70f, 1.0f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.06f, 0.08f, 0.14f, 1.0f));
        ImGui::PlotLines("##hist", fhist.data(), (int)fhist.size(),
                         0, nullptr, 0.0f, 1.0f, ImVec2(-1, Scaled(80)));
        ImGui::PopStyleColor(2);
    }
}

static void draw_event_log() {
    for (auto& ev : g_app.events) {
        ImVec4 col = {0.50f, 0.52f, 0.58f, 1.0f};
        if (ev.type == "CREATE")       col = {0.30f, 0.90f, 0.50f, 1.0f};
        else if (ev.type == "LOAD")    col = {0.30f, 0.80f, 1.00f, 1.0f};
        else if (ev.type == "SAVE")    col = {1.00f, 0.85f, 0.25f, 1.0f};
        else if (ev.type == "DELETE")  col = {1.00f, 0.35f, 0.35f, 1.0f};
        else if (ev.type == "TRAIN")   col = {0.50f, 0.70f, 1.00f, 1.0f};
        else if (ev.type == "ERROR")   col = {1.00f, 0.25f, 0.25f, 1.0f};
        else if (ev.type == "SYSTEM")  col = {0.60f, 0.55f, 0.80f, 1.0f};

        ImGui::TextColored({0.40f, 0.42f, 0.48f, 1.0f}, "%s", ev.time.c_str());
        ImGui::SameLine();
        ImGui::TextColored(col, "%-8s", ev.type.c_str());
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.78f, 0.85f, 1.0f));
        ImGui::TextWrapped("%s", ev.message.c_str());
        ImGui::PopStyleColor();
    }
}

// Forward declarations
static void stop_supervised_training();
static void apply_auto_config();

// Apply dataset-driven auto-configuration to the UI state and live network.
// Safe to call whenever a dataset is loaded — no-op when train_engine is null.
static void apply_auto_config() {
    if (!g_app.train_engine || !g_app.dataset_loaded) return;

    auto profile = training::profile_dataset(
        *g_app.train_engine, g_app.is_mnist,
        g_app.dataset_type == 2 ? g_app.image_target_w : 0,
        g_app.dataset_type == 2 ? g_app.image_target_h : 0);

    auto cfg = training::auto_supervised_config(profile);
    auto topo_overlay = training::auto_topology_config(profile);

    // Mirror SupervisedConfig into UI state so the user can review/override.
    g_app.sup_epochs = cfg.epochs;
    g_app.sup_batch_size = static_cast<int>(cfg.batch_size);
    g_app.sup_lr = static_cast<float>(cfg.optimizer_config.learning_rate);
    g_app.sup_optimizer = (cfg.optimizer_name == "adam") ? 1 : 0;
    g_app.sup_loss = static_cast<int>(cfg.loss_type);
    g_app.sup_warmup = cfg.warmup_epochs;
    g_app.sup_dropout = static_cast<float>(cfg.dropout_rate);
    g_app.sup_batch_norm = cfg.batch_norm;
    g_app.sup_augmentation = cfg.augmentation.enabled;
    g_app.sup_interleave = cfg.interleave_self_org;
    g_app.sup_self_org_interval = cfg.self_org_interval;
    if (cfg.schedule_name == "constant")         g_app.sup_schedule = 0;
    else if (cfg.schedule_name == "cosine")      g_app.sup_schedule = 1;
    else if (cfg.schedule_name == "exponential") g_app.sup_schedule = 2;

    // Apply topology overlay to the live network.
    auto topo = g_app.network.topology_config();
    topo.enable_neuron_pruning = topo_overlay.enable_neuron_pruning;
    topo.enable_neuron_growth = topo_overlay.enable_neuron_growth;
    topo.enable_dynamic_rewiring = topo_overlay.enable_dynamic_rewiring;
    topo.use_stall_counter = topo_overlay.use_stall_counter;
    topo.growth_patience = topo_overlay.growth_patience;
    topo.growth_split_ratio = topo_overlay.growth_split_ratio;
    topo.max_hidden_neurons = topo_overlay.max_hidden_neurons;
    g_app.network.set_topology_config(topo);

    g_app.sup_use_stall_counter = topo.use_stall_counter;
    g_app.sup_growth_patience = topo.growth_patience;

    add_event("AUTO", "Profile: " + training::describe_profile(profile));
    add_event("AUTO", "Config: " + std::to_string(cfg.epochs) + " epochs, batch " +
                      std::to_string(cfg.batch_size) + ", lr " +
                      std::to_string(cfg.optimizer_config.learning_rate) +
                      ", max_hidden=" + std::to_string(topo.max_hidden_neurons));
}

// =================================================================
// Dataset Panel
// =================================================================
// Helper to clean up previous dataset and load new one
static void do_load_dataset() {
    if (g_app.sup_training.load()) stop_supervised_training();

    // Remove old IO neurons
    if (g_app.train_engine && g_app.train_engine->io_manager()) {
        auto* iom = g_app.train_engine->io_manager();
        for (auto& id : iom->input_ids()) g_app.network.remove_neuron(id);
        for (auto& id : iom->output_ids()) g_app.network.remove_neuron(id);
    }

    g_app.train_engine = std::make_unique<training::TrainingEngine>(g_app.network);

    util::Result<void> result = util::Error("No dataset type selected");

    if (g_app.dataset_type == 0) {
        // CSV
        result = g_app.train_engine->load_dataset(
            g_app.dataset_path, g_app.num_target_cols, g_app.dataset_has_header);
        g_app.is_mnist = false;
    } else if (g_app.dataset_type == 1) {
        // MNIST IDX
        result = g_app.train_engine->load_mnist(
            g_app.mnist_images_path, g_app.mnist_labels_path, g_app.mnist_max_samples);
        g_app.is_mnist = true;
    } else {
        // Image Folder
        result = g_app.train_engine->load_image_folder(
            g_app.image_folder_path, g_app.image_target_w, g_app.image_target_h,
            g_app.mnist_max_samples);
        g_app.is_mnist = false;
    }

    if (result.ok()) {
        g_app.dataset_loaded = true;
        g_app.dataset_status = "Loaded: " +
            std::to_string(g_app.train_engine->num_samples()) + " samples, " +
            std::to_string(g_app.train_engine->num_features()) + " features, " +
            std::to_string(g_app.train_engine->num_outputs()) + " outputs";
        add_event("DATA", g_app.dataset_status);
        {
            std::lock_guard<std::mutex> lock(g_app.history_mutex);
            g_app.loss_history.clear();
            g_app.accuracy_history.clear();
            g_app.val_loss_history.clear();
            g_app.val_accuracy_history.clear();
        }
        // Auto-pick hyperparameters for this dataset unless the user opted out.
        if (g_app.auto_configure_on_load) {
            apply_auto_config();
        }
    } else {
        g_app.dataset_loaded = false;
        g_app.is_mnist = false;
        g_app.dataset_status = "Error: " + result.error().message;
        add_event("ERROR", g_app.dataset_status);
    }
}

static void draw_dataset_panel() {
    SectionHeader("DATA FORMAT");

    static const char* format_names[] = { "CSV", "MNIST (IDX)", "Image Folder" };
    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("##format", &g_app.dataset_type, format_names, 3);

    ImGui::Spacing();

    if (g_app.dataset_type == 0) {
        // CSV mode
        SectionHeader("CSV FILE");
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "File Path:");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##csvpath", g_app.dataset_path, sizeof(g_app.dataset_path));
        ImGui::Spacing();
        ImGui::SetNextItemWidth(Scaled(100));
        ImGui::InputInt("Target Cols", &g_app.num_target_cols);
        g_app.num_target_cols = std::max(1, g_app.num_target_cols);
        ImGui::Checkbox("Has Header Row", &g_app.dataset_has_header);
    } else if (g_app.dataset_type == 1) {
        // MNIST mode
        SectionHeader("MNIST FILES");
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Images file (idx3-ubyte):");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##mnistimg", g_app.mnist_images_path, sizeof(g_app.mnist_images_path));
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Labels file (idx1-ubyte):");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##mnistlbl", g_app.mnist_labels_path, sizeof(g_app.mnist_labels_path));
        ImGui::Spacing();
        ImGui::SetNextItemWidth(Scaled(120));
        ImGui::InputInt("Max Samples", &g_app.mnist_max_samples);
        g_app.mnist_max_samples = std::clamp(g_app.mnist_max_samples, 100, 60000);
        ImGui::TextColored(ImVec4(0.45f, 0.48f, 0.55f, 1.0f), "Use fewer for faster training");
    } else {
        // Image Folder mode
        SectionHeader("IMAGE FOLDER");
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Root folder:");
        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("##imgfolder", g_app.image_folder_path, sizeof(g_app.image_folder_path));
        ImGui::TextColored(ImVec4(0.45f, 0.48f, 0.55f, 1.0f), "Subfolders = classes, .bmp files");
        ImGui::Spacing();
        ImGui::SetNextItemWidth(Scaled(80));
        ImGui::InputInt("Width##imgw", &g_app.image_target_w);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(Scaled(80));
        ImGui::InputInt("Height##imgh", &g_app.image_target_h);
        g_app.image_target_w = std::clamp(g_app.image_target_w, 4, 256);
        g_app.image_target_h = std::clamp(g_app.image_target_h, 4, 256);
        ImGui::Spacing();
        ImGui::SetNextItemWidth(Scaled(120));
        ImGui::InputInt("Max Samples##img", &g_app.mnist_max_samples);
        g_app.mnist_max_samples = std::clamp(g_app.mnist_max_samples, 10, 100000);
    }

    ImGui::Spacing();
    ImGui::Checkbox("Auto-Configure on Load", &g_app.auto_configure_on_load);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("When ON: pick sensible defaults for epochs, batch size,\n"
                          "LR, schedule, dropout, augmentation, and adaptive\n"
                          "topology limits right after loading, so you can hit\n"
                          "Start Training immediately.\n"
                          "Use the Auto Configure button to re-run manually.");
    }

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.47f, 0.87f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.55f, 0.95f, 1.0f));
    if (ImGui::Button("Load Dataset", ImVec2(-1, Scaled(32)))) {
        do_load_dataset();
    }
    ImGui::PopStyleColor(2);

    if (!g_app.dataset_status.empty()) {
        ImGui::Spacing();
        ImVec4 col = g_app.dataset_loaded
            ? ImVec4(0.30f, 0.85f, 0.50f, 1.0f)
            : ImVec4(1.0f, 0.35f, 0.35f, 1.0f);
        ImGui::TextColored(col, "%s", g_app.dataset_status.c_str());
    }

    if (g_app.dataset_loaded && g_app.train_engine) {
        SectionHeader("DATASET INFO");
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Train samples");
        ImGui::SameLine(Scaled(130));
        ImGui::Text("%d", g_app.train_engine->num_train_samples());

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Test samples");
        ImGui::SameLine(Scaled(130));
        ImGui::Text("%d", g_app.train_engine->num_test_samples());

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Features");
        ImGui::SameLine(Scaled(130));
        ImGui::Text("%d", g_app.train_engine->num_features());

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Outputs");
        ImGui::SameLine(Scaled(130));
        ImGui::Text("%d", g_app.train_engine->num_outputs());

        auto& labels = g_app.train_engine->class_labels();
        if (!labels.empty()) {
            ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Classes");
            ImGui::SameLine(Scaled(130));
            std::string label_str;
            for (size_t i = 0; i < labels.size() && i < 8; ++i) {
                if (i > 0) label_str += ", ";
                label_str += labels[i];
            }
            if (labels.size() > 8) label_str += "...";
            ImGui::TextWrapped("%s", label_str.c_str());
        }
    }
}

// =================================================================
// Supervised Training Panel
// =================================================================
static void stop_supervised_training() {
    if (g_app.train_engine) g_app.train_engine->request_stop();
    if (g_app.sup_train_thread.joinable()) g_app.sup_train_thread.join();
    g_app.sup_training = false;
}

static void start_supervised_training() {
    if (g_app.sup_training.load() || !g_app.train_engine || !g_app.dataset_loaded) return;

    // Make sure previous thread is fully joined
    if (g_app.sup_train_thread.joinable()) g_app.sup_train_thread.join();

    g_app.sup_training = true;
    {
        std::lock_guard<std::mutex> lock(g_app.history_mutex);
        g_app.loss_history.clear();
        g_app.accuracy_history.clear();
        g_app.val_loss_history.clear();
        g_app.val_accuracy_history.clear();
    }

    static const char* schedule_keys[] = { "constant", "cosine", "exponential" };

    training::SupervisedConfig config;
    config.epochs = g_app.sup_epochs;
    config.batch_size = g_app.sup_batch_size;
    config.loss_type = static_cast<training::LossType>(g_app.sup_loss);
    config.optimizer_name = (g_app.sup_optimizer == 1) ? "adam" : "sgd";
    config.optimizer_config.learning_rate = g_app.sup_lr;
    config.interleave_self_org = g_app.sup_interleave;
    config.self_org_interval = g_app.sup_self_org_interval;

    // Push adaptive-topology toggles into the live network config so self-org
    // honors the user's choices on the very next cycle.
    {
        auto topo = g_app.network.topology_config();
        topo.use_stall_counter = g_app.sup_use_stall_counter;
        topo.growth_patience = g_app.sup_growth_patience;
        g_app.network.set_topology_config(topo);
    }
    // Phase 1 features
    config.schedule_name = schedule_keys[g_app.sup_schedule];
    config.warmup_epochs = g_app.sup_warmup;
    config.dropout_rate = g_app.sup_dropout;
    config.batch_norm = g_app.sup_batch_norm;
    config.augmentation.enabled = g_app.sup_augmentation;
    if (g_app.is_mnist) {
        config.augmentation.image_width = 28;
        config.augmentation.image_height = 28;
    }

    add_event("TRAIN", "Supervised training started (" +
              std::to_string(config.epochs) + " epochs, " +
              config.optimizer_name + ", lr=" +
              std::to_string(config.optimizer_config.learning_rate) + ")");

    g_app.sup_train_thread = std::thread([config]() {
        g_app.train_engine->train(config, [](const training::EpochResult& r) {
            std::lock_guard<std::mutex> lock(g_app.history_mutex);
            g_app.loss_history.push_back(static_cast<float>(r.train_loss));
            g_app.accuracy_history.push_back(static_cast<float>(r.train_accuracy));
            g_app.val_loss_history.push_back(static_cast<float>(r.val_loss));
            g_app.val_accuracy_history.push_back(static_cast<float>(r.val_accuracy));
        });
        g_app.sup_training = false;
        add_event("TRAIN", "Supervised training completed");
    });
}

static void draw_supervised_panel() {
    static const char* optimizer_names[] = { "SGD", "Adam" };
    static const char* loss_names[] = { "MSE", "Cross-Entropy" };
    static const char* schedule_names[] = { "Constant", "Cosine Annealing", "Exponential Decay" };

    SectionHeader("CONFIGURATION");

    ImGui::SetNextItemWidth(Scaled(100));
    ImGui::InputInt("Epochs", &g_app.sup_epochs);
    g_app.sup_epochs = std::clamp(g_app.sup_epochs, 1, 10000);

    ImGui::SetNextItemWidth(Scaled(100));
    ImGui::InputInt("Batch Size", &g_app.sup_batch_size);
    g_app.sup_batch_size = std::clamp(g_app.sup_batch_size, 1, 10000);

    ImGui::SetNextItemWidth(-1);
    ImGui::SliderFloat("Learning Rate##sup", &g_app.sup_lr, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);

    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("Optimizer", &g_app.sup_optimizer, optimizer_names, 2);

    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("Loss Function", &g_app.sup_loss, loss_names, 2);

    SectionHeader("REGULARIZATION");

    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("LR Schedule", &g_app.sup_schedule, schedule_names, 3);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Cosine: smoothly decays LR to near-zero\n"
                          "Exponential: multiplies LR by 0.95 each epoch\n"
                          "Constant: no decay (not recommended)");
    }

    ImGui::SetNextItemWidth(Scaled(100));
    ImGui::InputInt("Warmup Epochs", &g_app.sup_warmup);
    g_app.sup_warmup = std::clamp(g_app.sup_warmup, 0, g_app.sup_epochs / 2);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Linearly ramp LR from 0 to initial value\n"
                          "over this many epochs. Prevents early instability.");
    }

    ImGui::SetNextItemWidth(-1);
    ImGui::SliderFloat("Dropout", &g_app.sup_dropout, 0.0f, 0.8f, "%.2f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Randomly drops neuron activations during training\n"
                          "to prevent overfitting. 0.2 = 20%% drop rate.\n"
                          "Set to 0 to disable.");
    }

    ImGui::Checkbox("Batch Normalization", &g_app.sup_batch_norm);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Normalizes neuron pre-activations to stabilize\n"
                          "gradients and allow higher learning rates.\n"
                          "Speeds up convergence significantly.");
    }

    ImGui::Checkbox("Data Augmentation", &g_app.sup_augmentation);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Applies random transformations to training images:\n"
                          "rotation, shift, noise, flip.\n"
                          "Multiplies effective dataset size 5-10x.");
    }

    ImGui::Spacing();
    ImGui::Checkbox("Self-Organization", &g_app.sup_interleave);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Enables adaptive topology: Hebbian wiring,\n"
                          "neuron growth/pruning, dynamic rewiring.\n"
                          "The network will optimize its own structure.");
    }
    if (g_app.sup_interleave) {
        ImGui::SetNextItemWidth(Scaled(100));
        ImGui::InputInt("Every N Epochs", &g_app.sup_self_org_interval);
        g_app.sup_self_org_interval = std::clamp(g_app.sup_self_org_interval, 1, 1000);

        ImGui::Checkbox("Gate Growth on Loss Plateau", &g_app.sup_use_stall_counter);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("When ON: grow only after training loss stalls for N cycles.\n"
                              "When OFF: grow every self-org cycle (bounded by split ratio).\n"
                              "Turn OFF for aggressive continuous expansion or when\n"
                              "no supervised loss is being reported.");
        }
        if (g_app.sup_use_stall_counter) {
            ImGui::SetNextItemWidth(Scaled(100));
            ImGui::InputInt("Growth Patience", &g_app.sup_growth_patience);
            g_app.sup_growth_patience = std::clamp(g_app.sup_growth_patience, 1, 1000);
        }
    }

    SectionHeader("CONTROLS");

    if (g_app.sup_training) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.18f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.88f, 0.22f, 0.22f, 1.0f));
        if (ImGui::Button("Stop Training##sup", ImVec2(-1, Scaled(36)))) {
            stop_supervised_training();
        }
        ImGui::PopStyleColor(2);

        if (g_app.train_engine) {
            ImGui::Spacing();
            float progress = static_cast<float>(g_app.train_engine->current_epoch()) /
                             std::max(1, g_app.sup_epochs);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.20f, 0.65f, 0.90f, 1.0f));
            char overlay[64];
            snprintf(overlay, sizeof(overlay), "Epoch %d / %d",
                     g_app.train_engine->current_epoch(), g_app.sup_epochs);
            ImGui::ProgressBar(progress, ImVec2(-1, Scaled(20)), overlay);
            ImGui::PopStyleColor();
        }
    } else {
        bool can_train = g_app.dataset_loaded && g_app.train_engine && !g_app.network.neurons().empty();
        if (!can_train) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.22f, 0.28f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.42f, 0.48f, 1.0f));
            ImGui::Button("Start Training##sup", ImVec2(-1, Scaled(36)));
            ImGui::PopStyleColor(2);
            if (!g_app.dataset_loaded)
                ImGui::TextColored(ImVec4(0.50f, 0.42f, 0.20f, 1.0f), "Load a dataset first");
            else if (g_app.network.neurons().empty())
                ImGui::TextColored(ImVec4(0.50f, 0.42f, 0.20f, 1.0f), "Create a network first");
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.47f, 0.87f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.55f, 0.95f, 1.0f));
            if (ImGui::Button("Start Training##sup", ImVec2(-1, Scaled(36)))) {
                start_supervised_training();
            }
            ImGui::PopStyleColor(2);

            // LR Finder button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.45f, 0.30f, 0.65f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.55f, 0.38f, 0.75f, 1.0f));
            if (ImGui::Button("Find Best LR##lrfind", ImVec2(-1, Scaled(28)))) {
                add_event("TRAIN", "Running LR finder...");
                auto lr_result = g_app.train_engine->find_learning_rate();
                g_app.sup_lr = static_cast<float>(lr_result.suggested_lr);
                add_event("TRAIN", "Suggested LR: " + std::to_string(lr_result.suggested_lr));
            }
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Sweeps learning rate from 1e-7 to 1.0,\n"
                                  "finds the rate where loss drops fastest,\n"
                                  "then auto-sets the LR slider.");
            }

            // Auto Configure button — one-click dataset-driven tuning.
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.60f, 0.40f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.72f, 0.48f, 1.0f));
            if (ImGui::Button("Auto Configure##autocfg", ImVec2(-1, Scaled(28)))) {
                apply_auto_config();
            }
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Analyzes the loaded dataset and picks sensible defaults:\n"
                                  "  epochs, batch size, LR, optimizer, schedule, warmup,\n"
                                  "  dropout, batch-norm, augmentation, self-org interval,\n"
                                  "  and adaptive topology cap.\n"
                                  "Tuned for fast convergence on this dataset's size/shape.");
            }
        }
    }

    // Training curves — snapshot under lock, then render
    std::vector<float> snap_loss, snap_acc, snap_vloss, snap_vacc;
    {
        std::lock_guard<std::mutex> lock(g_app.history_mutex);
        snap_loss = g_app.loss_history;
        snap_acc = g_app.accuracy_history;
        snap_vloss = g_app.val_loss_history;
        snap_vacc = g_app.val_accuracy_history;
    }

    if (!snap_loss.empty()) {
        SectionHeader("TRAINING CURVES");

        ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(1.0f, 0.45f, 0.20f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.06f, 0.08f, 0.14f, 1.0f));
        ImGui::PlotLines("##loss", snap_loss.data(), (int)snap_loss.size(), 0,
                         "Train Loss", 0.0f, FLT_MAX, ImVec2(-1, Scaled(60)));
        ImGui::PopStyleColor(2);

        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.30f, 0.90f, 0.50f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.06f, 0.08f, 0.14f, 1.0f));
        ImGui::PlotLines("##acc", snap_acc.data(), (int)snap_acc.size(), 0,
                         "Train Accuracy", 0.0f, 1.0f, ImVec2(-1, Scaled(60)));
        ImGui::PopStyleColor(2);

        if (!snap_vloss.empty()) {
            ImGui::Spacing();

            ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.90f, 0.70f, 0.20f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.06f, 0.08f, 0.14f, 1.0f));
            ImGui::PlotLines("##valloss", snap_vloss.data(), (int)snap_vloss.size(), 0,
                             "Val Loss", 0.0f, FLT_MAX, ImVec2(-1, Scaled(60)));
            ImGui::PopStyleColor(2);

            ImGui::Spacing();

            ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.50f, 0.80f, 1.0f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.06f, 0.08f, 0.14f, 1.0f));
            ImGui::PlotLines("##valacc", snap_vacc.data(), (int)snap_vacc.size(), 0,
                             "Val Accuracy", 0.0f, 1.0f, ImVec2(-1, Scaled(60)));
            ImGui::PopStyleColor(2);
        }

        // Latest metrics
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Last Train Loss");
        ImGui::SameLine(Scaled(140));
        ImGui::Text("%.6f", snap_loss.back());

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Last Train Acc");
        ImGui::SameLine(Scaled(140));
        ImGui::Text("%.2f%%", snap_acc.back() * 100.0f);

        if (!snap_vacc.empty()) {
            ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Last Val Acc");
            ImGui::SameLine(Scaled(140));
            ImGui::Text("%.2f%%", snap_vacc.back() * 100.0f);
        }

        // Export buttons
        SectionHeader("EXPORT METRICS");

        {
            static char export_status[256] = "";
            static float export_status_timer = 0.0f;

            float btn_w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.50f, 0.35f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.20f, 0.60f, 0.42f, 1.0f));
            if (ImGui::Button("Export CSV", ImVec2(btn_w, Scaled(30)))) {
                auto& hist = g_app.train_engine->history();
                if (!hist.empty()) {
                    auto name = training::MetricsExporter::timestamped_name();
                    std::string fpath = "./metrics/" + name + ".csv";
                    training::MetricsExporter::export_csv(fpath, hist);
                    snprintf(export_status, sizeof(export_status), "Saved %s", fpath.c_str());
                    export_status_timer = 4.0f;
                    add_event("EXPORT", std::string("CSV exported: ") + fpath);
                }
            }
            ImGui::PopStyleColor(2);

            ImGui::SameLine();

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.50f, 0.35f, 0.15f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.60f, 0.42f, 0.20f, 1.0f));
            if (ImGui::Button("Export JSON", ImVec2(btn_w, Scaled(30)))) {
                auto& hist = g_app.train_engine->history();
                if (!hist.empty()) {
                    auto name = training::MetricsExporter::timestamped_name();
                    std::string fpath = "./metrics/" + name + ".json";
                    training::MetricsExporter::export_json(fpath, hist);
                    snprintf(export_status, sizeof(export_status), "Saved %s", fpath.c_str());
                    export_status_timer = 4.0f;
                    add_event("EXPORT", std::string("JSON exported: ") + fpath);
                }
            }
            ImGui::PopStyleColor(2);

            // Show status message with fade-out
            if (export_status_timer > 0.0f) {
                float alpha = std::min(export_status_timer, 1.0f);
                ImGui::TextColored(ImVec4(0.30f, 0.85f, 0.50f, alpha), "%s", export_status);
                export_status_timer -= ImGui::GetIO().DeltaTime;
            }
        }
    }

    // Adaptive topology stats
    if (g_app.sup_interleave) {
        SectionHeader("ADAPTIVE TOPOLOGY");
        auto& stats = g_app.network.adaptive_stats();

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Neurons Pruned");
        ImGui::SameLine(Scaled(150));
        ImGui::Text("%d", stats.neurons_pruned_total);

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Neurons Grown");
        ImGui::SameLine(Scaled(150));
        ImGui::Text("%d", stats.neurons_grown_total);

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Synapses Rewired");
        ImGui::SameLine(Scaled(150));
        ImGui::Text("%d", stats.synapses_rewired_total);

        const auto& topo = g_app.network.topology_config();
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Stall Counter");
        ImGui::SameLine(Scaled(150));
        if (topo.use_stall_counter) {
            ImGui::Text("%d / %d", stats.cycles_since_improvement,
                        topo.growth_patience);
        } else {
            ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "disabled");
        }

        // Count hidden neurons
        int hidden = 0;
        if (g_app.train_engine && g_app.train_engine->io_manager()) {
            auto& prot = g_app.network.protected_neurons();
            for (auto& [id, _] : g_app.network.neurons()) {
                if (!prot.count(id)) hidden++;
            }
        }
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Hidden Neurons");
        ImGui::SameLine(Scaled(150));
        ImGui::Text("%d", hidden);

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Total Synapses");
        ImGui::SameLine(Scaled(150));
        ImGui::Text("%zu", g_app.network.synapse_count());
    }
}

// =================================================================
// Inference Panel
// =================================================================
static void do_predict_from_input() {
    std::vector<double> features;
    std::string input(g_app.inference_input);
    std::stringstream ss(input);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            size_t start = token.find_first_not_of(" \t");
            if (start != std::string::npos)
                features.push_back(std::stod(token.substr(start)));
        } catch (...) {}
    }
    if (!features.empty()) {
        // Save image for MNIST visualization
        if (g_app.is_mnist && features.size() == 784) {
            g_app.last_mnist_image.resize(784);
            for (int i = 0; i < 784; ++i)
                g_app.last_mnist_image[i] = static_cast<float>(features[i]);
        }
        g_app.last_prediction = g_app.train_engine->predict(features);
        g_app.last_predicted_class = g_app.train_engine->predict_class(features);
        g_app.last_predicted_label = g_app.train_engine->predict_label(features);
        add_event("INFER", "Predicted: " + g_app.last_predicted_label);
    }
}

static void do_predict_random_test_sample() {
    if (!g_app.train_engine) return;
    auto& history = g_app.train_engine->history();
    // Get a random test sample from the split
    // We need access to the raw data — use the engine's predict on raw features
    // For now, pick from the training data (we stored all_data_)
    // Actually the training engine doesn't expose raw samples, so we'll use
    // the DataLoader to re-read a random one. For MNIST, we can just grab from the
    // existing loaded data through the forward pass.

    // Simpler approach: load one random image from the IDX file
    if (g_app.is_mnist && strlen(g_app.mnist_images_path) > 0) {
        std::ifstream img_file(g_app.mnist_images_path, std::ios::binary);
        std::ifstream lbl_file(g_app.mnist_labels_path, std::ios::binary);
        if (img_file && lbl_file) {
            // Skip headers
            img_file.seekg(16);
            lbl_file.seekg(8);

            // Pick random sample
            static std::mt19937 rng(static_cast<unsigned>(
                std::chrono::system_clock::now().time_since_epoch().count()));
            int max_idx = std::min(g_app.mnist_max_samples, 10000);
            int idx = std::uniform_int_distribution<int>(0, max_idx - 1)(rng);

            img_file.seekg(16 + idx * 784);
            lbl_file.seekg(8 + idx);

            std::vector<uint8_t> pixels(784);
            img_file.read(reinterpret_cast<char*>(pixels.data()), 784);
            uint8_t label;
            lbl_file.read(reinterpret_cast<char*>(&label), 1);

            // Store for visualization
            g_app.last_mnist_image.resize(784);
            std::vector<double> features(784);
            for (int i = 0; i < 784; ++i) {
                features[i] = static_cast<double>(pixels[i]) / 255.0;
                g_app.last_mnist_image[i] = static_cast<float>(features[i]);
            }

            g_app.last_prediction = g_app.train_engine->predict(features);
            g_app.last_predicted_class = g_app.train_engine->predict_class(features);
            g_app.last_predicted_label = g_app.train_engine->predict_label(features);
            add_event("INFER", "Sample #" + std::to_string(idx) +
                      " (true=" + std::to_string((int)label) +
                      ") predicted=" + g_app.last_predicted_label);
        }
    }
}

static void draw_inference_panel() {
    bool can_predict = g_app.train_engine && g_app.dataset_loaded;

    if (g_app.is_mnist && can_predict) {
        // MNIST-specific: random test sample button
        SectionHeader("MNIST TEST");
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.50f, 0.16f, 0.70f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.60f, 0.22f, 0.82f, 1.0f));
        if (ImGui::Button("Test Random Digit", ImVec2(-1, Scaled(36)))) {
            do_predict_random_test_sample();
        }
        ImGui::PopStyleColor(2);

        // Draw the digit image if we have one
        if (!g_app.last_mnist_image.empty()) {
            ImGui::Spacing();

            // Render 28x28 digit as a scaled-up pixel grid
            float pixel_size = Scaled(6);
            float img_w = 28.0f * pixel_size;
            float img_h = 28.0f * pixel_size;

            // Center the image
            float avail = ImGui::GetContentRegionAvail().x;
            float offset_x = (avail - img_w) * 0.5f;
            if (offset_x > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);

            ImVec2 p = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();

            // Background
            dl->AddRectFilled(p, ImVec2(p.x + img_w, p.y + img_h), IM_COL32(5, 8, 15, 255));

            for (int y = 0; y < 28; ++y) {
                for (int x = 0; x < 28; ++x) {
                    float v = g_app.last_mnist_image[y * 28 + x];
                    if (v > 0.01f) {
                        int intensity = static_cast<int>(v * 255);
                        // Blue-white gradient for digit pixels
                        int r = static_cast<int>(v * 180 + 75 * v);
                        int g = static_cast<int>(v * 200 + 55 * v);
                        int b = static_cast<int>(255);
                        dl->AddRectFilled(
                            ImVec2(p.x + x * pixel_size, p.y + y * pixel_size),
                            ImVec2(p.x + (x + 1) * pixel_size - 1, p.y + (y + 1) * pixel_size - 1),
                            IM_COL32(r, g, b, intensity));
                    }
                }
            }
            // Reserve space
            ImGui::Dummy(ImVec2(img_w, img_h));
        }
    }

    SectionHeader("MANUAL INPUT");
    ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Comma-separated values:");
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##infer_input", g_app.inference_input, sizeof(g_app.inference_input));

    ImGui::Spacing();
    if (!can_predict) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.22f, 0.28f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.42f, 0.48f, 1.0f));
        ImGui::Button("Predict", ImVec2(-1, Scaled(32)));
        ImGui::PopStyleColor(2);
        ImGui::TextColored(ImVec4(0.50f, 0.42f, 0.20f, 1.0f), "Load data & train first");
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.50f, 0.16f, 0.70f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.60f, 0.22f, 0.82f, 1.0f));
        if (ImGui::Button("Predict", ImVec2(-1, Scaled(32)))) {
            do_predict_from_input();
        }
        ImGui::PopStyleColor(2);
    }

    if (!g_app.last_prediction.empty() && g_app.train_engine) {
        SectionHeader("RESULT");

        // Large predicted digit display for MNIST
        if (g_app.is_mnist) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.30f, 0.90f, 0.50f, 1.0f));
            float avail = ImGui::GetContentRegionAvail().x;
            char digit_str[8];
            snprintf(digit_str, sizeof(digit_str), "%s", g_app.last_predicted_label.c_str());
            ImVec2 text_size = ImGui::CalcTextSize(digit_str);
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - text_size.x) * 0.5f);
            ImGui::Text("Predicted Digit: %s", digit_str);
            ImGui::PopStyleColor();
        } else {
            ImGui::TextColored(ImVec4(0.30f, 0.90f, 0.50f, 1.0f), "Predicted: %s",
                               g_app.last_predicted_label.c_str());
        }
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Class index: %d",
                           g_app.last_predicted_class);

        SectionHeader("CONFIDENCE");

        auto probs = training::softmax(g_app.last_prediction);
        auto& labels = g_app.train_engine->class_labels();
        for (size_t i = 0; i < probs.size(); ++i) {
            std::string lbl = (i < labels.size()) ? labels[i] : ("out_" + std::to_string(i));
            bool is_winner = (static_cast<int>(i) == g_app.last_predicted_class);
            ImVec4 bar_col = is_winner
                ? ImVec4(0.30f, 0.90f, 0.50f, 1.0f)
                : ImVec4(0.20f, 0.45f, 0.80f, 1.0f);

            ImGui::TextColored(ImVec4(0.75f, 0.78f, 0.85f, 1.0f), "%-4s", lbl.c_str());
            ImGui::SameLine(Scaled(40));
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_col);
            char pct[16];
            snprintf(pct, sizeof(pct), "%.1f%%", probs[i] * 100.0);
            ImGui::ProgressBar(static_cast<float>(probs[i]), ImVec2(-1, Scaled(16)), pct);
            ImGui::PopStyleColor();
        }
    }
}

// =================================================================
// Create Network Dialog
// =================================================================
static void draw_create_dialog() {
    if (!g_app.show_create_dialog) return;

    ImGui::OpenPopup("Create New Network");
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(Scaled(480, 420));

    if (ImGui::BeginPopupModal("Create New Network", &g_app.show_create_dialog)) {
        ImGui::Spacing();
        ImGui::Text("Configure your new neural network:");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("Name", g_app.new_name, sizeof(g_app.new_name));
        ImGui::Spacing();

        ImGui::SetNextItemWidth(Scaled(180));
        ImGui::InputInt("Neurons", &g_app.new_neurons);
        ImGui::SetNextItemWidth(Scaled(180));
        ImGui::InputInt("Connections", &g_app.new_connections);
        ImGui::Spacing();

        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("Task Type", &g_app.new_task_type, task_type_names, 6);
        ImGui::SetNextItemWidth(Scaled(180));
        ImGui::InputFloat("Learning Rate", &g_app.new_lr, 0.001f, 0.01f, "%.4f");
        ImGui::Spacing();

        ImGui::InputTextMultiline("Description", g_app.new_desc, sizeof(g_app.new_desc),
                                  ImVec2(-1, Scaled(70)));

        g_app.new_neurons = std::clamp(g_app.new_neurons, 2, 50000);
        g_app.new_connections = std::clamp(g_app.new_connections, 0, 200000);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Buttons aligned right
        float btn_w = Scaled(140);
        float cancel_w = Scaled(90);
        ImGui::SetCursorPosX(ImGui::GetContentRegionAvail().x - btn_w - cancel_w - Scaled(8));

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.47f, 0.87f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.55f, 0.95f, 1.0f));
        if (ImGui::Button("Create & Load", ImVec2(btn_w, Scaled(34)))) {
            create_network();
            g_app.show_create_dialog = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleColor(2);
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(cancel_w, Scaled(34)))) {
            g_app.show_create_dialog = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

static void draw_delete_dialog() {
    if (!g_app.show_delete_dialog) return;

    ImGui::OpenPopup("Confirm Deletion");
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(Scaled(380, 160));

    if (ImGui::BeginPopupModal("Confirm Deletion", &g_app.show_delete_dialog)) {
        ImGui::Spacing();
        ImGui::Text("Delete \"%s\"?", g_app.delete_target.c_str());
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "This action cannot be undone.");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float btn_w = Scaled(90);
        ImGui::SetCursorPosX(ImGui::GetContentRegionAvail().x - btn_w * 2 - Scaled(8));

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.70f, 0.15f, 0.15f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.20f, 0.20f, 1.0f));
        if (ImGui::Button("Delete", ImVec2(btn_w, Scaled(32)))) {
            delete_image(g_app.delete_target);
            g_app.show_delete_dialog = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::PopStyleColor(2);
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(btn_w, Scaled(32)))) {
            g_app.show_delete_dialog = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

// =================================================================
// Initial Docking Layout
// =================================================================
static void setup_initial_layout(ImGuiID dockspace_id) {
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->WorkSize);

    // Split: left sidebar (20%) | center+right (80%)
    ImGuiID dock_left, dock_main_right;
    ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.20f,
                                 &dock_left, &dock_main_right);

    // Split center+right: center (70%) | right sidebar (30%)
    ImGuiID dock_center, dock_right;
    ImGui::DockBuilderSplitNode(dock_main_right, ImGuiDir_Right, 0.28f,
                                 &dock_right, &dock_center);

    // Split center: viewport (75%) | bottom log (25%)
    ImGuiID dock_viewport, dock_bottom;
    ImGui::DockBuilderSplitNode(dock_center, ImGuiDir_Down, 0.22f,
                                 &dock_bottom, &dock_viewport);

    // Split right sidebar: top panels | bottom panels
    ImGuiID dock_right_top, dock_right_bottom;
    ImGui::DockBuilderSplitNode(dock_right, ImGuiDir_Down, 0.50f,
                                 &dock_right_bottom, &dock_right_top);

    // Split right top: inspector | stats
    ImGuiID dock_inspector, dock_stats;
    ImGui::DockBuilderSplitNode(dock_right_top, ImGuiDir_Down, 0.50f,
                                 &dock_stats, &dock_inspector);

    // Split left sidebar: files (top) | dataset (bottom)
    ImGuiID dock_left_top, dock_left_bottom;
    ImGui::DockBuilderSplitNode(dock_left, ImGuiDir_Down, 0.45f,
                                 &dock_left_bottom, &dock_left_top);

    // Assign windows to docks
    ImGui::DockBuilderDockWindow("Neural Images", dock_left_top);
    ImGui::DockBuilderDockWindow("Dataset", dock_left_bottom);
    ImGui::DockBuilderDockWindow("3D Network View", dock_viewport);
    ImGui::DockBuilderDockWindow("Event Log", dock_bottom);
    ImGui::DockBuilderDockWindow("Inspector", dock_inspector);
    ImGui::DockBuilderDockWindow("Network Stats", dock_stats);
    ImGui::DockBuilderDockWindow("Training", dock_right_bottom);
    ImGui::DockBuilderDockWindow("Supervised Training", dock_right_bottom);
    ImGui::DockBuilderDockWindow("Save Network", dock_right_bottom);
    ImGui::DockBuilderDockWindow("Inference", dock_right_bottom);

    ImGui::DockBuilderFinish(dockspace_id);
}

// =================================================================
// Main Frame
// =================================================================
static void draw_frame() {
    // P1-11 — License gate. Two cases:
    //  * `modal_block` is set when the boot probe found no valid
    //    license AND no live trial. In that case the dialog is the
    //    only thing on screen and we return early -- no wizard, no
    //    dockspace, no menu bar. Activating a key clears the flag
    //    and we fall through to normal rendering on the next frame.
    //  * `open` (without modal_block) means the user clicked
    //    File > License...; the dialog floats over whatever's
    //    underneath and the rest of the UI keeps rendering.
    if (g_app.license_dialog.modal_block) {
        workbench::render_license_dialog(g_app.license_dialog,
                                         g_app.license_status);
        return;
    }

    // P1-09 — Wizard short-circuit. When wizard_mode is on, the
    // wizard owns the whole viewport; we skip the legacy dockspace +
    // panels entirely. The wizard returns true on the frame the user
    // taps "Show advanced UI", which flips us back to the old layout
    // (and forces a layout reset so the dockspace re-initializes
    // cleanly after frames of being absent).
    if (g_app.wizard_mode) {
        if (!g_wizard_runtime) {
            g_wizard_runtime = workbench::make_wizard_runtime();
        }
        if (workbench::render_wizard(g_wizard_state, *g_wizard_runtime)) {
            g_app.wizard_mode  = false;
            g_app.first_frame  = true;  // re-init dockspace next frame
        }
        return;
    }

    // Dockspace
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags host_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_MenuBar;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("DockSpaceHost", nullptr, host_flags);
    ImGui::PopStyleVar(3);

    // Menu bar
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Network", "Ctrl+N")) g_app.show_create_dialog = true;
            if (ImGui::MenuItem("Refresh Models")) refresh_registry();
            ImGui::Separator();
            // P1-11 — re-open the license dialog so users can upgrade
            // Trial -> paid mid-session, or sign out.
            if (ImGui::MenuItem("License...")) {
                workbench::open_license_dialog(g_app.license_dialog);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) PostQuitMessage(0);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Reset Layout")) g_app.first_frame = true;
            // P1-09 — way back to the wizard from the advanced UI.
            if (ImGui::MenuItem("Show guided wizard")) g_app.wizard_mode = true;
            ImGui::EndMenu();
        }

        // Right-aligned status
        float status_w = Scaled(300);
        ImGui::SameLine(ImGui::GetWindowWidth() - status_w);
        if (!g_app.loaded_image_name.empty()) {
            ImGui::TextColored(ImVec4(0.30f, 0.85f, 0.50f, 1.0f), "Active: %s",
                               g_app.loaded_image_name.c_str());
        } else {
            ImGui::TextColored(ImVec4(0.50f, 0.52f, 0.58f, 1.0f), "No network loaded");
        }

        ImGui::EndMenuBar();
    }

    ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");

    // Setup initial layout on first run (or reset)
    if (g_app.first_frame) {
        setup_initial_layout(dockspace_id);
        g_app.first_frame = false;
    }

    ImGui::DockSpace(dockspace_id, ImVec2(0, 0), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();

    // --- File Manager Panel ---
    ImGui::Begin("Neural Images");
    draw_file_panel();
    ImGui::End();

    // --- 3D Network View ---
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("3D Network View");
    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("canvas", canvas_size,
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    draw_3d_view(canvas_pos, canvas_size);
    ImGui::End();
    ImGui::PopStyleVar();

    // --- Training Controls ---
    ImGui::Begin("Training");
    {
        bool is_training = g_app.training.load();

        SectionHeader("CONTROLS");

        if (is_training) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.18f, 0.18f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.88f, 0.22f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.65f, 0.14f, 0.14f, 1.0f));
            if (ImGui::Button("Stop Training", ImVec2(-1, Scaled(38)))) stop_training();
            ImGui::PopStyleColor(3);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.47f, 0.87f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.55f, 0.95f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.12f, 0.40f, 0.78f, 1.0f));
            if (ImGui::Button("Start Training", ImVec2(-1, Scaled(38)))) start_training();
            ImGui::PopStyleColor(3);
        }

        ImGui::Spacing();
        if (ImGui::Button("Step Once", ImVec2(-1, Scaled(30)))) {
            g_app.network.self_organize(1);
            add_event("TRAIN", "Step " + std::to_string(g_app.network.stats().organization_cycle));
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Steps: %d", g_app.train_steps_done);
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Cycle:  %d",
                           (int)g_app.network.stats().organization_cycle);

        SectionHeader("PARAMETERS");

        if (ImGui::SliderFloat("Learn Rate", &g_app.learning_rate, 0.001f, 0.1f, "%.4f")) {
            auto tc = g_app.network.topology_config();
            tc.learning_rate = g_app.learning_rate;
            g_app.network.set_topology_config(tc);
        }
        if (ImGui::SliderFloat("Affinity", &g_app.affinity_threshold, 0.05f, 1.0f, "%.2f")) {
            auto tc = g_app.network.topology_config();
            tc.affinity_threshold = g_app.affinity_threshold;
            g_app.network.set_topology_config(tc);
        }
        ImGui::SliderInt("Interval", &g_app.stream_interval_ms, 10, 1000, "%d ms");
    }
    ImGui::End();

    // --- Save Panel ---
    ImGui::Begin("Save Network");
    {
        SectionHeader("SAVE AS");

        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("Name##save", g_app.save_name, sizeof(g_app.save_name));
        ImGui::Spacing();

        ImGui::SetNextItemWidth(-1);
        ImGui::InputTextMultiline("Desc##save", g_app.save_desc, sizeof(g_app.save_desc),
                                  ImVec2(-1, Scaled(55)));

        ImGui::SetNextItemWidth(-1);
        ImGui::Combo("Task##save", &g_app.save_task_type, task_type_names, 6);
        ImGui::Spacing();

        ImGui::SetNextItemWidth(-1);
        ImGui::InputText("Tags##save", g_app.save_tags, sizeof(g_app.save_tags));
        ImGui::TextColored(ImVec4(0.45f, 0.48f, 0.55f, 1.0f), "Comma-separated tags");

        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.47f, 0.87f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.55f, 0.95f, 1.0f));
        if (ImGui::Button("Save as New", ImVec2(-1, Scaled(32)))) {
            if (strlen(g_app.save_name) > 0) save_image(g_app.save_name);
        }
        ImGui::PopStyleColor(2);

        if (!g_app.loaded_image_name.empty()) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.50f, 0.38f, 0.12f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.65f, 0.48f, 0.15f, 1.0f));
            if (ImGui::Button(("Overwrite: " + g_app.loaded_image_name).c_str(),
                              ImVec2(-1, Scaled(32)))) {
                save_image(g_app.loaded_image_name, true);
            }
            ImGui::PopStyleColor(2);
        }
    }
    ImGui::End();

    // --- Inspector ---
    ImGui::Begin("Inspector");
    draw_inspector();
    ImGui::End();

    // --- Stats ---
    ImGui::Begin("Network Stats");
    {
        SectionHeader("OVERVIEW");

        auto stats = g_app.network.stats();

        // Stat rows with labels + values
        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Neurons");
        ImGui::SameLine(Scaled(130));
        ImGui::Text("%d", (int)stats.neuron_count);

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Synapses");
        ImGui::SameLine(Scaled(130));
        ImGui::Text("%d", (int)stats.synapse_count);

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Avg Activation");
        ImGui::SameLine(Scaled(130));
        ImGui::Text("%.6f", stats.avg_activation);

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Avg Weight");
        ImGui::SameLine(Scaled(130));
        ImGui::Text("%.6f", stats.avg_weight);

        ImGui::TextColored(ImVec4(0.55f, 0.58f, 0.65f, 1.0f), "Org Cycle");
        ImGui::SameLine(Scaled(130));
        ImGui::Text("%d", (int)stats.organization_cycle);

        // Track history
        static int frame_counter = 0;
        if (++frame_counter % 30 == 0) {
            g_app.activation_history.push_back(static_cast<float>(stats.avg_activation));
            g_app.weight_history.push_back(static_cast<float>(stats.avg_weight));
            g_app.neuron_count_history.push_back(static_cast<float>(stats.neuron_count));
            if (g_app.activation_history.size() > 200) {
                g_app.activation_history.erase(g_app.activation_history.begin());
                g_app.weight_history.erase(g_app.weight_history.begin());
                g_app.neuron_count_history.erase(g_app.neuron_count_history.begin());
            }
        }

        if (!g_app.activation_history.empty()) {
            SectionHeader("CHARTS");

            ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.30f, 0.70f, 1.0f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.06f, 0.08f, 0.14f, 1.0f));
            ImGui::PlotLines("##act", g_app.activation_history.data(),
                             (int)g_app.activation_history.size(), 0,
                             "Activation", 0.0f, 1.0f, ImVec2(-1, Scaled(60)));
            ImGui::PopStyleColor(2);

            ImGui::Spacing();

            ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.90f, 0.60f, 0.20f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.06f, 0.08f, 0.14f, 1.0f));
            ImGui::PlotLines("##wt", g_app.weight_history.data(),
                             (int)g_app.weight_history.size(), 0,
                             "Weight", 0.0f, 1.0f, ImVec2(-1, Scaled(60)));
            ImGui::PopStyleColor(2);
        }

        if (!g_app.loaded_image_name.empty()) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.30f, 0.85f, 0.50f, 1.0f), "Active: %s",
                               g_app.loaded_image_name.c_str());
        }
    }
    ImGui::End();

    // --- Event Log ---
    ImGui::Begin("Event Log");
    ImGui::BeginChild("EventLogScroll", ImVec2(0, 0), ImGuiChildFlags_None);
    draw_event_log();
    ImGui::EndChild();
    ImGui::End();

    // --- Dataset Panel ---
    ImGui::Begin("Dataset");
    draw_dataset_panel();
    ImGui::End();

    // --- Supervised Training Panel ---
    ImGui::Begin("Supervised Training");
    draw_supervised_panel();
    ImGui::End();

    // --- Inference Panel ---
    ImGui::Begin("Inference");
    draw_inference_panel();
    ImGui::End();

    // Dialogs
    draw_create_dialog();
    draw_delete_dialog();

    // P1-11 — License dialog rendered as a floating modal over the
    // advanced UI. Only renders if the user opened it from
    // File > License...; the modal_block path is handled at the top
    // of draw_frame() before any other UI is drawn.
    workbench::render_license_dialog(g_app.license_dialog,
                                     g_app.license_status);
}

// =================================================================
// Style Setup
// =================================================================
static void setup_style() {
    ImGui::StyleColorsDark();
    ImGuiStyle& s = ImGui::GetStyle();

    // Geometry - scaled by DPI
    s.WindowRounding    = Scaled(6);
    s.FrameRounding     = Scaled(4);
    s.GrabRounding      = Scaled(4);
    s.TabRounding       = Scaled(4);
    s.ScrollbarRounding = Scaled(6);
    s.ChildRounding     = Scaled(4);
    s.PopupRounding     = Scaled(4);

    s.WindowPadding     = Scaled(12, 12);
    s.FramePadding      = Scaled(10, 6);
    s.ItemSpacing       = Scaled(10, 8);
    s.ItemInnerSpacing  = Scaled(8, 6);
    s.IndentSpacing     = Scaled(20);
    s.ScrollbarSize     = Scaled(14);
    s.GrabMinSize       = Scaled(12);
    s.WindowBorderSize  = 1.0f;
    s.FrameBorderSize   = 0.0f;
    s.TabBorderSize     = 0.0f;
    s.SeparatorTextBorderSize = Scaled(2);

    // Color palette - high contrast dark theme with blue accents
    ImVec4* c = s.Colors;

    // Backgrounds
    c[ImGuiCol_WindowBg]             = ImVec4(0.06f, 0.07f, 0.11f, 1.00f);
    c[ImGuiCol_ChildBg]              = ImVec4(0.06f, 0.07f, 0.11f, 0.00f);
    c[ImGuiCol_PopupBg]              = ImVec4(0.08f, 0.09f, 0.14f, 0.97f);

    // Borders
    c[ImGuiCol_Border]               = ImVec4(0.18f, 0.20f, 0.28f, 0.80f);
    c[ImGuiCol_BorderShadow]         = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame (input fields, etc)
    c[ImGuiCol_FrameBg]              = ImVec4(0.10f, 0.12f, 0.18f, 1.00f);
    c[ImGuiCol_FrameBgHovered]       = ImVec4(0.14f, 0.16f, 0.24f, 1.00f);
    c[ImGuiCol_FrameBgActive]        = ImVec4(0.16f, 0.20f, 0.30f, 1.00f);

    // Title bars
    c[ImGuiCol_TitleBg]              = ImVec4(0.05f, 0.06f, 0.09f, 1.00f);
    c[ImGuiCol_TitleBgActive]        = ImVec4(0.08f, 0.10f, 0.16f, 1.00f);
    c[ImGuiCol_TitleBgCollapsed]     = ImVec4(0.04f, 0.05f, 0.08f, 0.80f);

    // Menu bar
    c[ImGuiCol_MenuBarBg]            = ImVec4(0.07f, 0.08f, 0.12f, 1.00f);

    // Scrollbar
    c[ImGuiCol_ScrollbarBg]          = ImVec4(0.05f, 0.06f, 0.09f, 0.60f);
    c[ImGuiCol_ScrollbarGrab]        = ImVec4(0.20f, 0.24f, 0.35f, 1.00f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.28f, 0.32f, 0.45f, 1.00f);
    c[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.34f, 0.40f, 0.55f, 1.00f);

    // Buttons
    c[ImGuiCol_Button]               = ImVec4(0.14f, 0.18f, 0.28f, 1.00f);
    c[ImGuiCol_ButtonHovered]        = ImVec4(0.20f, 0.26f, 0.40f, 1.00f);
    c[ImGuiCol_ButtonActive]         = ImVec4(0.16f, 0.44f, 0.82f, 1.00f);

    // Headers (collapsing headers, tree nodes, selectable)
    c[ImGuiCol_Header]               = ImVec4(0.12f, 0.16f, 0.26f, 1.00f);
    c[ImGuiCol_HeaderHovered]        = ImVec4(0.18f, 0.24f, 0.38f, 1.00f);
    c[ImGuiCol_HeaderActive]         = ImVec4(0.16f, 0.44f, 0.82f, 1.00f);

    // Separator
    c[ImGuiCol_Separator]            = ImVec4(0.18f, 0.20f, 0.28f, 1.00f);
    c[ImGuiCol_SeparatorHovered]     = ImVec4(0.30f, 0.50f, 0.80f, 0.80f);
    c[ImGuiCol_SeparatorActive]      = ImVec4(0.30f, 0.50f, 0.80f, 1.00f);

    // Resize grip
    c[ImGuiCol_ResizeGrip]           = ImVec4(0.20f, 0.30f, 0.50f, 0.40f);
    c[ImGuiCol_ResizeGripHovered]    = ImVec4(0.30f, 0.50f, 0.80f, 0.60f);
    c[ImGuiCol_ResizeGripActive]     = ImVec4(0.30f, 0.50f, 0.80f, 0.90f);

    // Tabs
    c[ImGuiCol_Tab]                  = ImVec4(0.08f, 0.10f, 0.16f, 1.00f);
    c[ImGuiCol_TabHovered]           = ImVec4(0.16f, 0.44f, 0.82f, 0.80f);
    c[ImGuiCol_TabSelected]          = ImVec4(0.14f, 0.36f, 0.68f, 1.00f);
    c[ImGuiCol_TabDimmed]            = ImVec4(0.06f, 0.08f, 0.12f, 1.00f);
    c[ImGuiCol_TabDimmedSelected]    = ImVec4(0.10f, 0.22f, 0.42f, 1.00f);

    // Docking
    c[ImGuiCol_DockingPreview]       = ImVec4(0.16f, 0.44f, 0.82f, 0.70f);
    c[ImGuiCol_DockingEmptyBg]       = ImVec4(0.04f, 0.05f, 0.08f, 1.00f);

    // Slider
    c[ImGuiCol_SliderGrab]           = ImVec4(0.25f, 0.55f, 0.95f, 1.00f);
    c[ImGuiCol_SliderGrabActive]     = ImVec4(0.35f, 0.65f, 1.00f, 1.00f);

    // Check mark
    c[ImGuiCol_CheckMark]            = ImVec4(0.30f, 0.65f, 1.00f, 1.00f);

    // Plot
    c[ImGuiCol_PlotLines]            = ImVec4(0.40f, 0.70f, 1.00f, 1.00f);
    c[ImGuiCol_PlotLinesHovered]     = ImVec4(0.50f, 0.80f, 1.00f, 1.00f);
    c[ImGuiCol_PlotHistogram]        = ImVec4(0.25f, 0.55f, 0.95f, 1.00f);
    c[ImGuiCol_PlotHistogramHovered] = ImVec4(0.35f, 0.65f, 1.00f, 1.00f);

    // Text
    c[ImGuiCol_Text]                 = ImVec4(0.85f, 0.88f, 0.95f, 1.00f);
    c[ImGuiCol_TextDisabled]         = ImVec4(0.40f, 0.42f, 0.50f, 1.00f);
    c[ImGuiCol_TextSelectedBg]       = ImVec4(0.16f, 0.44f, 0.82f, 0.40f);

    // Nav
    c[ImGuiCol_NavHighlight]         = ImVec4(0.30f, 0.60f, 1.00f, 1.00f);
}

// =================================================================
// DirectX 11 Helpers
// =================================================================
static bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT flags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_0 };
    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        flags, levels, 1, D3D11_SDK_VERSION, &sd,
        &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (hr == DXGI_ERROR_UNSUPPORTED)
        hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr,
            flags, levels, 1, D3D11_SDK_VERSION, &sd,
            &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (FAILED(hr)) return false;

    CreateRenderTarget();
    return true;
}

static void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

static void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    if (pBackBuffer) {
        g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
        pBackBuffer->Release();
    }
}

static void CleanupRenderTarget() {
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED) {
            g_ResizeWidth = LOWORD(lParam);
            g_ResizeHeight = HIWORD(lParam);
        }
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

// =================================================================
// Main
// =================================================================
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    // Enable per-monitor DPI awareness
    SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);

    // Get DPI scale for primary monitor
    HMONITOR monitor = MonitorFromPoint({0, 0}, MONITOR_DEFAULTTOPRIMARY);
    UINT dpiX = 96, dpiY = 96;
    GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &dpiX, &dpiY);
    g_dpiScale = dpiX / 96.0f;

    // Scale window size by DPI
    int winW = static_cast<int>(1600 * g_dpiScale);
    int winH = static_cast<int>(900 * g_dpiScale);

    // Create application window
    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L,
        GetModuleHandle(nullptr), nullptr, nullptr, nullptr, nullptr,
        L"NeuroGraphWorkbench", nullptr };
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowW(wc.lpszClassName, L"NeuroGraph Workbench",
        WS_OVERLAPPEDWINDOW, 100, 50, winW, winH, nullptr, nullptr, wc.hInstance, nullptr);

    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.IniFilename = nullptr; // Disable ini persistence — we set layout programmatically

    // Font: scale to DPI. Use built-in ProggyClean at large size, or the default at scaled size
    float fontSize = 16.0f * g_dpiScale;
    ImFontConfig fontConfig;
    fontConfig.OversampleH = 2;
    fontConfig.OversampleV = 2;
    fontConfig.PixelSnapH = true;
    io.Fonts->AddFontDefault(&fontConfig);
    // Scale the default font
    io.FontGlobalScale = fontSize / 13.0f;

    // Apply our custom theme
    setup_style();

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // Initialize NeuroGraph
    std::filesystem::create_directories(g_app.models_dir);
    refresh_registry();
    add_event("SYSTEM", "NeuroGraph Workbench started");
    add_event("SYSTEM", std::to_string(g_app.registry.count()) + " neural images found in ./models");

    // P1-11 — Boot-time license probe. If there's no activated key
    // and no live trial, raise the modal block so draw_frame()
    // shows nothing but the license dialog until the user enters a
    // valid key. The probe itself is in neurograph_lib so it's unit-
    // tested without touching D3D11.
    g_app.license_status = workbench::boot_license_state();
    if (!g_app.license_status.license) {
        g_app.license_dialog.modal_block = true;
        g_app.license_dialog.open        = true;
        if (g_app.license_status.trial_days_left < 0) {
            add_event("LICENSE", "No active license and trial has ended -- entry required");
        } else {
            add_event("LICENSE", "License state could not be determined -- entry required");
        }
    } else if (g_app.license_status.is_trial) {
        add_event("LICENSE",
            "Trial active: " +
            std::to_string(g_app.license_status.trial_days_left) +
            " day(s) remaining");
    } else {
        add_event("LICENSE",
            "Activated license loaded for " +
            g_app.license_status.license->holder);
    }

    // Main loop
    bool done = false;
    while (!done) {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done) break;

        // Handle resize
        if (g_ResizeWidth != 0 && g_ResizeHeight != 0) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight,
                                        DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        // Frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        draw_frame();

        ImGui::Render();
        const float clear_color[4] = { 0.03f, 0.04f, 0.07f, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        g_pSwapChain->Present(1, 0); // vsync
    }

    // Cleanup
    stop_training();
    stop_supervised_training();

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);

    return 0;
}
