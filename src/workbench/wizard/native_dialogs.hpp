#pragma once

// Native file dialogs and shell helpers for the wizard (Windows). On other
// platforms the pickers return nothing and the user types a path instead.

#include <filesystem>
#include <optional>
#include <string>

namespace neurograph::workbench {

/// "Open" dialog for a CSV file. Empty when cancelled.
std::optional<std::filesystem::path> pick_csv_file();

/// "Save as" dialog for a CSV file, pre-filled with `default_name`.
std::optional<std::filesystem::path> pick_save_csv(const std::filesystem::path& default_name);

/// Folder picker. Empty when cancelled.
std::optional<std::filesystem::path> pick_folder(const wchar_t* title);

/// Open a file or folder with its default application (Explorer for a
/// folder). Text files with no registered application open in Notepad.
void open_with_shell(const std::filesystem::path& p);

/// UTF-8 text of a path, for display in the UI.
std::string path_utf8(const std::filesystem::path& p);

/// Path from UTF-8 text typed into the UI.
std::filesystem::path path_from_utf8(const char* text);

} // namespace neurograph::workbench
