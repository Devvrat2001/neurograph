#pragma once

// P1-09 — CSV peek utility for the wizard's Load step.
//
// We only need the column names, a row count, and the first few data
// rows to populate the preview table. Full CSV parsing (with type
// inference, missing-value counts, quoted-comma handling) happens
// later in the orchestrator via the dataqc scanner + pipeline fit.
//
// Kept as a separate translation unit from wizard_state so unit tests
// can mock `CsvMetadata` directly without opening real files — the
// transitions in wizard_state never call this; the workbench UI does.

#include "neurograph/util/result.hpp"
#include "neurograph/workbench/wizard_state.hpp"

#include <cstddef>
#include <filesystem>

namespace neurograph::workbench {

/// Read the header + up to `preview_rows` data rows from a CSV file,
/// and count the remaining rows with a streaming pass. Returns a
/// `CsvMetadata` suitable for `set_csv`. Errors go through
/// util::Result with human-readable messages (the wizard shows them
/// as "Couldn't open file: <message>").
///
/// Parsing is permissive-but-simple: treats commas as separators,
/// handles `""` inside double-quoted fields, strips a trailing `\r`.
/// Good enough for the preview; the real parser in the orchestrator
/// handles the harder edge cases.
util::Result<CsvMetadata>
peek_csv_metadata(const std::filesystem::path& path,
                  std::size_t                  preview_rows = 5);

} // namespace neurograph::workbench
