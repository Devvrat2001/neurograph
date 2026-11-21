#include "native_dialogs.hpp"

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <shellapi.h>
#  include <shobjidl.h>
#endif

#include <string>

namespace neurograph::workbench {

namespace fs = std::filesystem;

std::string path_utf8(const fs::path& p) {
    const auto u = p.u8string();
    return std::string(u.begin(), u.end());
}

fs::path path_from_utf8(const char* text) {
    return fs::path(reinterpret_cast<const char8_t*>(text));
}

#ifdef _WIN32

namespace {

// The file dialogs are COM objects and need a single-threaded apartment
// on the calling (UI) thread.
struct ComScope {
    HRESULT hr;
    ComScope() : hr(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)) {}
    ~ComScope() { if (SUCCEEDED(hr)) CoUninitialize(); }
};

// Show the dialog and return the chosen file-system path, if any.
std::optional<fs::path> run_dialog(IFileDialog* dlg) {
    if (FAILED(dlg->Show(GetActiveWindow()))) return std::nullopt;   // also on Cancel
    IShellItem* item = nullptr;
    if (FAILED(dlg->GetResult(&item)) || !item) return std::nullopt;
    std::optional<fs::path> out;
    PWSTR name = nullptr;
    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &name)) && name) {
        out = fs::path(name);
        CoTaskMemFree(name);
    }
    item->Release();
    return out;
}

} // namespace

std::optional<fs::path> pick_csv_file() {
    ComScope com;
    IFileOpenDialog* dlg = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dlg)))) {
        return std::nullopt;
    }
    const COMDLG_FILTERSPEC types[] = {
        {L"CSV files (*.csv, *.txt)", L"*.csv;*.txt"},
        {L"All files", L"*.*"},
    };
    dlg->SetFileTypes(2, types);
    dlg->SetTitle(L"Choose a CSV file");
    auto out = run_dialog(dlg);
    dlg->Release();
    return out;
}

std::optional<fs::path> pick_save_csv(const fs::path& default_name) {
    ComScope com;
    IFileSaveDialog* dlg = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dlg)))) {
        return std::nullopt;
    }
    const COMDLG_FILTERSPEC types[] = {{L"CSV file", L"*.csv"}};
    dlg->SetFileTypes(1, types);
    dlg->SetDefaultExtension(L"csv");
    dlg->SetFileName(default_name.filename().wstring().c_str());
    dlg->SetTitle(L"Save the predictions as");
    auto out = run_dialog(dlg);
    dlg->Release();
    return out;
}

std::optional<fs::path> pick_folder(const wchar_t* title) {
    ComScope com;
    IFileOpenDialog* dlg = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dlg)))) {
        return std::nullopt;
    }
    DWORD flags = 0;
    dlg->GetOptions(&flags);
    dlg->SetOptions(flags | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
    dlg->SetTitle(title);
    auto out = run_dialog(dlg);
    dlg->Release();
    return out;
}

void open_with_shell(const fs::path& p) {
    const auto r = reinterpret_cast<INT_PTR>(
        ShellExecuteW(nullptr, L"open", p.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    if (r <= 32 && fs::is_regular_file(p)) {
        // No application registered for this file type (.md, typically).
        ShellExecuteW(nullptr, L"open", L"notepad.exe",
                      (L"\"" + p.wstring() + L"\"").c_str(), nullptr, SW_SHOWNORMAL);
    }
}

#else

std::optional<fs::path> pick_csv_file() { return std::nullopt; }
std::optional<fs::path> pick_save_csv(const fs::path&) { return std::nullopt; }
std::optional<fs::path> pick_folder(const wchar_t*) { return std::nullopt; }
void open_with_shell(const fs::path&) {}

#endif

} // namespace neurograph::workbench
