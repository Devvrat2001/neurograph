// Draws the workbench wizard off-screen and saves each step as a PNG.
//
// The pictures come from the app's own step renderers, fonts and colours,
// drawn by the same Direct3D 11 backend into a texture instead of a
// window, so they show what a user would see without starting the app.
// A real model is trained on the given file for the Results pictures.
//
//   neurograph_wizard_snapshots --csv data.csv [--out dir] [--target col]
//       [--prefix name-] [--size 1584x861] [--dpi 1.0] [--hardware]
//
// --size is the window's client area; 1584x861 is the workbench's
// default 1600x900 window at 100% scaling. The software rasteriser
// (WARP) is used unless --hardware is given, so the output is the same
// on every machine.

#include "native_dialogs.hpp"
#include "theme.hpp"
#include "wizard_ui.hpp"
#include "neurograph/orchestration/auto_pipeline.hpp"
#include "neurograph/workbench/csv_preview.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include "imgui.h"
#include "imgui_impl_dx11.h"

#include <d3d11.h>
#include <objbase.h>
#include <wincodec.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace neurograph;
using workbench::TrainingStatus;
using workbench::WizardState;
using workbench::WizardStep;

namespace {

template <typename T>
void release(T*& p) {
    if (p) { p->Release(); p = nullptr; }
}

struct Options {
    fs::path    csv;
    fs::path    out = "snapshots";
    std::string target;
    std::string prefix;
    int         width    = 1584;
    int         height   = 861;
    float       dpi      = 1.0f;
    bool        hardware = false;
};

bool write_png(const fs::path& path, const std::vector<std::uint8_t>& bgr,
               UINT w, UINT h) {
    IWICImagingFactory*    factory = nullptr;
    IWICStream*            stream  = nullptr;
    IWICBitmapEncoder*     encoder = nullptr;
    IWICBitmapFrameEncode* frame   = nullptr;
    IPropertyBag2*         props   = nullptr;
    WICPixelFormatGUID     format  = GUID_WICPixelFormat24bppBGR;

    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                  CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (SUCCEEDED(hr)) hr = factory->CreateStream(&stream);
    if (SUCCEEDED(hr)) hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE);
    if (SUCCEEDED(hr)) hr = factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder);
    if (SUCCEEDED(hr)) hr = encoder->Initialize(stream, WICBitmapEncoderNoCache);
    if (SUCCEEDED(hr)) hr = encoder->CreateNewFrame(&frame, &props);
    if (SUCCEEDED(hr)) hr = frame->Initialize(props);
    if (SUCCEEDED(hr)) hr = frame->SetSize(w, h);
    if (SUCCEEDED(hr)) hr = frame->SetPixelFormat(&format);
    if (SUCCEEDED(hr) && format != GUID_WICPixelFormat24bppBGR) hr = E_FAIL;
    if (SUCCEEDED(hr)) {
        hr = frame->WritePixels(h, w * 3, static_cast<UINT>(bgr.size()),
                                const_cast<BYTE*>(bgr.data()));
    }
    if (SUCCEEDED(hr)) hr = frame->Commit();
    if (SUCCEEDED(hr)) hr = encoder->Commit();

    release(props);
    release(frame);
    release(encoder);
    release(stream);
    release(factory);
    return SUCCEEDED(hr);
}

// A texture ImGui draws into, and a CPU-readable copy of it.
class Canvas {
public:
    ~Canvas() {
        release_targets();
        release(context_);
        release(device_);
    }

    bool init(bool hardware) {
        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0};
        const D3D_DRIVER_TYPE driver = hardware ? D3D_DRIVER_TYPE_HARDWARE
                                                : D3D_DRIVER_TYPE_WARP;
        return SUCCEEDED(D3D11CreateDevice(nullptr, driver, nullptr, 0, levels, 1,
                                           D3D11_SDK_VERSION, &device_, nullptr,
                                           &context_));
    }

    bool resize(int w, int h) {
        release_targets();
        width_  = w;
        height_ = h;
        D3D11_TEXTURE2D_DESC td{};
        td.Width            = static_cast<UINT>(w);
        td.Height           = static_cast<UINT>(h);
        td.MipLevels        = 1;
        td.ArraySize        = 1;
        td.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;   // as the app's swap chain
        td.SampleDesc.Count = 1;
        td.Usage            = D3D11_USAGE_DEFAULT;
        td.BindFlags        = D3D11_BIND_RENDER_TARGET;
        if (FAILED(device_->CreateTexture2D(&td, nullptr, &target_))) return false;
        if (FAILED(device_->CreateRenderTargetView(target_, nullptr, &view_))) return false;
        td.Usage          = D3D11_USAGE_STAGING;
        td.BindFlags      = 0;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        return SUCCEEDED(device_->CreateTexture2D(&td, nullptr, &staging_));
    }

    ID3D11Device*        device()  const { return device_; }
    ID3D11DeviceContext* context() const { return context_; }
    int width()  const { return width_; }
    int height() const { return height_; }

    void begin() {
        // Same clear colour as the workbench's main loop.
        const float clear[4] = {0.03f, 0.04f, 0.07f, 1.0f};
        context_->OMSetRenderTargets(1, &view_, nullptr);
        context_->ClearRenderTargetView(view_, clear);
    }

    bool save(const fs::path& path) {
        context_->CopyResource(staging_, target_);
        D3D11_MAPPED_SUBRESOURCE m{};
        if (FAILED(context_->Map(staging_, 0, D3D11_MAP_READ, 0, &m))) return false;
        std::vector<std::uint8_t> bgr(static_cast<std::size_t>(width_) * height_ * 3);
        for (int y = 0; y < height_; ++y) {
            const auto* row = static_cast<const std::uint8_t*>(m.pData) +
                              static_cast<std::size_t>(y) * m.RowPitch;
            std::uint8_t* dst = bgr.data() + static_cast<std::size_t>(y) * width_ * 3;
            for (int x = 0; x < width_; ++x) {
                dst[x * 3 + 0] = row[x * 4 + 2];
                dst[x * 3 + 1] = row[x * 4 + 1];
                dst[x * 3 + 2] = row[x * 4 + 0];
            }
        }
        context_->Unmap(staging_, 0);
        return write_png(path, bgr, static_cast<UINT>(width_), static_cast<UINT>(height_));
    }

private:
    void release_targets() {
        release(staging_);
        release(view_);
        release(target_);
    }

    ID3D11Device*           device_  = nullptr;
    ID3D11DeviceContext*    context_ = nullptr;
    ID3D11Texture2D*        target_  = nullptr;
    ID3D11RenderTargetView* view_    = nullptr;
    ID3D11Texture2D*        staging_ = nullptr;
    int width_  = 0;
    int height_ = 0;
};

class Snapshots {
public:
    Snapshots(Canvas& canvas, const Options& o) : canvas_(canvas), o_(o) {}

    // One wizard frame, exactly as the workbench's main loop draws it.
    void frame(WizardState& state, workbench::WizardRuntime& rt) {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(static_cast<float>(canvas_.width()),
                                static_cast<float>(canvas_.height()));
        io.DeltaTime   = 1.0f / 60.0f;
        ImGui_ImplDX11_NewFrame();
        ImGui::NewFrame();
        (void)workbench::render_wizard(state, rt);
        ImGui::Render();
        canvas_.begin();
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    }

    // A few frames so tables and wrapped text settle, then save the last.
    bool shot(WizardState& state, workbench::WizardRuntime& rt, const std::string& name) {
        for (int i = 0; i < 4; ++i) frame(state, rt);
        const fs::path file = o_.out / (o_.prefix + name + ".png");
        if (!canvas_.save(file)) {
            std::fprintf(stderr, "could not write %s\n", workbench::path_utf8(file).c_str());
            return false;
        }
        std::printf("wrote %s\n", workbench::path_utf8(file).c_str());
        return true;
    }

private:
    Canvas&        canvas_;
    const Options& o_;
};

util::Result<orchestration::AutoResult>
train(WizardState& state, std::vector<orchestration::Progress>* seen) {
    state.export_dir = fs::temp_directory_path() / "neurograph_snapshots_bundle";
    return orchestration::run(
        state.csv->path, state.target_column,
        workbench::build_options(state, state.export_dir),
        [seen](const orchestration::Progress& p) {
            if (seen) seen->push_back(p);
        },
        {});
}

// A progress update from the middle of model training, for the
// "training in progress" picture.
orchestration::Progress mid_training(const std::vector<orchestration::Progress>& seen) {
    for (const auto& p : seen) {
        if (p.stage == orchestration::Stage::Training && p.overall_fraction >= 0.3) return p;
    }
    return seen.empty() ? orchestration::Progress{} : seen[seen.size() / 2];
}

bool load(WizardState& state, const fs::path& csv) {
    auto meta = workbench::peek_csv_metadata(csv, 5);
    if (!meta.ok()) {
        std::fprintf(stderr, "could not read %s: %s\n", workbench::path_utf8(csv).c_str(),
                     meta.error().message.c_str());
        return false;
    }
    workbench::set_csv(state, std::move(meta).value());
    return true;
}

// The user's file, step by step: returns false when any step could not
// be shown or saved.
bool walk_through_file(Snapshots& snap, Canvas& canvas, const Options& o) {
    WizardState state;
    auto rt = workbench::make_wizard_runtime();
    bool ok = snap.shot(state, *rt, "1-load-empty");

    // Same route as dropping the file onto the window.
    rt->dropped_file = o.csv;
    snap.frame(state, *rt);
    if (!state.csv) {
        std::fprintf(stderr, "the wizard could not load the file: %s\n",
                     state.load_error.c_str());
        return false;
    }
    if (!o.target.empty() && !workbench::set_target_column(state, o.target)) {
        std::fprintf(stderr, "no column named %s\n", o.target.c_str());
        return false;
    }
    ok = snap.shot(state, *rt, "2-load") && ok;

    workbench::advance(state);
    ok = snap.shot(state, *rt, "3-quality") && ok;

    if (!workbench::advance(state)) {
        std::fprintf(stderr, "cannot go on to training: %s\n",
                     workbench::can_advance(state).c_str());
        return false;
    }
    ok = snap.shot(state, *rt, "4-train") && ok;

    std::vector<orchestration::Progress> seen;
    auto result = train(state, &seen);
    state.status.store(TrainingStatus::Running);
    workbench::on_progress(state, mid_training(seen));
    ok = snap.shot(state, *rt, "5-training") && ok;

    workbench::on_training_finished(state, std::move(result));
    if (state.status.load() != TrainingStatus::Succeeded) {
        snap.shot(state, *rt, "6-train-failed");
        std::fprintf(stderr, "training failed: %s\n", state.error_message.c_str());
        return false;
    }
    ok = snap.shot(state, *rt, "6-results") && ok;

    // The whole Results page at once, as if the window were very tall.
    const int h = canvas.height();
    ok = canvas.resize(canvas.width(), 2400) && snap.shot(state, *rt, "7-results-full") && ok;
    return canvas.resize(canvas.width(), h) && ok;
}

// A file that cannot be trained on, because every row has the same
// answer: the blocking check and the failed run.
bool walk_through_blocked_file(Snapshots& snap) {
    const fs::path tiny = fs::temp_directory_path() / "neurograph_snapshots_one_answer.csv";
    {
        std::ofstream f(tiny, std::ios::binary);
        f << "x,y,answer\n";
        for (int i = 1; i <= 8; ++i) f << i << ',' << (i * 7) % 5 << ",yes\n";
    }
    WizardState state;
    auto rt = workbench::make_wizard_runtime();
    bool ok = load(state, tiny);
    if (ok) {
        state.current_step = WizardStep::Quality;
        ok = snap.shot(state, *rt, "8-quality-blocked");

        state.current_step = WizardStep::Train;
        workbench::on_training_finished(state, train(state, nullptr));
        ok = snap.shot(state, *rt, "9-train-failed") && ok;
    }
    std::error_code ec;
    fs::remove(tiny, ec);
    return ok;
}

int run(const Options& o) {
    std::error_code ec;
    fs::create_directories(o.out, ec);

    Canvas canvas;
    if (!canvas.init(o.hardware) || !canvas.resize(o.width, o.height)) {
        std::fprintf(stderr, "could not create a Direct3D 11 device\n");
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags        |= ImGuiConfigFlags_DockingEnable;   // as the workbench
    io.IniFilename         = nullptr;
    io.LogFilename         = nullptr;
    io.BackendPlatformName = "wizard_snapshots";
    io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);                   // no hover effects
    workbench::load_fonts(o.dpi);
    workbench::apply_style(o.dpi);
    ImGui_ImplDX11_Init(canvas.device(), canvas.context());

    Snapshots snap(canvas, o);
    bool ok = walk_through_file(snap, canvas, o);
    ok = walk_through_blocked_file(snap) && ok;

    ImGui_ImplDX11_Shutdown();
    ImGui::DestroyContext();
    return ok ? 0 : 1;
}

std::string utf8(const wchar_t* w) {
    return workbench::path_utf8(fs::path(w));
}

void usage() {
    std::fprintf(stderr,
        "usage: neurograph_wizard_snapshots --csv FILE [--out DIR] [--target COLUMN]\n"
        "           [--prefix NAME] [--size WxH] [--dpi SCALE] [--hardware]\n");
}

} // namespace

int wmain(int argc, wchar_t** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::wstring a = argv[i];
        const bool has_value = i + 1 < argc;
        if (a == L"--csv" && has_value)         o.csv    = argv[++i];
        else if (a == L"--out" && has_value)    o.out    = argv[++i];
        else if (a == L"--target" && has_value) o.target = utf8(argv[++i]);
        else if (a == L"--prefix" && has_value) o.prefix = utf8(argv[++i]);
        else if (a == L"--dpi" && has_value)    o.dpi    = std::wcstof(argv[++i], nullptr);
        else if (a == L"--size" && has_value) {
            if (std::swscanf(argv[++i], L"%dx%d", &o.width, &o.height) != 2) { usage(); return 2; }
        }
        else if (a == L"--hardware")            o.hardware = true;
        else { usage(); return 2; }
    }
    if (o.csv.empty() || o.width < 320 || o.height < 240 || !(o.dpi > 0.0f)) {
        usage();
        return 2;
    }

    if (FAILED(CoInitializeEx(nullptr, COINIT_MULTITHREADED))) return 1;
    const int rc = run(o);
    CoUninitialize();
    return rc;
}
