# NeuroGraph

Train a prediction model from a CSV file on your own Windows PC: no Python,
no cloud service, no data leaving the machine. Then use the model from the
desktop app, the command line, or your own programs.

> **Status:** 0.2, pre-release, Windows only. 484 unit and integration
> tests and 12 performance tests pass on a fresh checkout; two more run
> only when the MNIST files are placed in `datasets/mnist`.

---

## What it does

1. **Checks the file.** Blank cells, identifier columns, constant columns,
   badly imbalanced classes, and columns that give the answer away. Only
   problems that make training impossible stop it; the rest are explained.
2. **Prepares the columns automatically.** Numbers (including `$1,200`,
   `45%`, `(1,234.50)`), categories, and dates (day count, month, weekday).
   Identifier and free-text columns are left out, and blank cells are
   filled and flagged. Everything is fitted on the training rows only.
3. **Trains and compares three kinds of model.** A straight-line model,
   gradient-boosted decision trees, and a neural network that widens its
   hidden layer while that keeps helping. The best one on held-back rows
   wins. A simpler model is kept when the others are not clearly better,
   and a blend of the two best is used when it clearly beats both.
4. **Reports honestly.** Scores come from rows that neither training nor
   model selection saw, shown next to what simple guessing would score.
   It also shows which columns the model relies on and whether more data
   would likely help.
5. **Saves a model folder.** The workbench ("Score a CSV file"),
   `neurograph_predict.exe` and `neurograph_rt.dll` (C API) all use it.
   A model report (`card.md`) and a manifest with file hashes are written
   alongside.

## Accuracy

Each dataset was split once into a training file and an untouched test
file. NeuroGraph trained on the training file with its default settings
("Balanced") and nothing removed by hand. Four scikit-learn 1.8 models
(logistic or ridge regression, random forest, histogram gradient
boosting, and an MLP) trained on the same file with standard
preprocessing. All were scored on the same test rows.

| Data (rows) | Predicting | NeuroGraph 0.2 | Best scikit-learn model | Training time |
|---|---|---|---|---|
| Iris (150) | 3 classes | 97.8% | 100% (MLP) | < 1 s |
| Wine (178) | 3 classes | 98.1% | 100% (several) | < 1 s |
| Breast cancer (569) | 2 classes | 93.0% | 97.1% (boosted trees) | < 1 s |
| Digits (1,797) | 10 classes | 96.5% | 98.2% (boosted trees) | 11 s |
| Customer churn (6,000), IDs and dates left in | 2 classes | 83.5%, AUC 0.80 | 84.3%, AUC 0.80 (boosted trees) | < 1 s |
| Synthetic (12,000 × 20) | 2 classes | 93.1% | 93.3% (MLP) | 4 s |
| Synthetic (60,000 × 40) | 2 classes | 94.7% | 95.2% (MLP) | 43 s |
| Diabetes (442) | a number | R² 0.394 | R² 0.400 (ridge) | < 1 s |
| Friedman #1 (5,000) | a number | R² 0.948 | R² 0.944 (MLP) | 2 s |

NeuroGraph is within about a point of the best of the four on most
datasets. It trails on small files because it holds back 30% of the
training file for its own checks, while the reference models trained on
all of it. The scripts that produce this table are in
[`tools/bench`](tools/bench).

## Programs

| Program | What it is for |
|---|---|
| `neurograph_workbench.exe` | Desktop app. Guided steps: load a CSV (Browse or drag and drop), review the check and the columns, train, see the results, then save the model or score another file. |
| `neurograph_train.exe` | Command-line trainer: CSV in, model folder out. Readable report or `--json`. |
| `neurograph_predict.exe` | Command-line scoring: model folder + CSV in, predictions out. Refuses files that lack a column the model needs, unless given `--allow-missing`. |
| `neurograph_rt.dll` | C API for scoring from C, C++, C#, Rust, Go, Python (`ctypes`) and others. |

### Command line

```
neurograph_train --data customers.csv --target churned --out churn_model
neurograph_predict --model churn_model --in new_customers.csv --out predictions.csv
```

`neurograph_train --help` lists the options: time allowance, columns to
leave out, forcing classification or regression, larger models, and a
fixed seed. The same file, options and seed give an identical model, as
long as no model runs into the time limit.

## How the rows are used

The file is split once, before anything is fitted:

- **70% train**: every candidate model learns from these rows.
- **15% check**: decides when each model stops and which model wins.
- **15% test**: used once, at the end, for the scores you are shown.

Rows with a blank target are skipped and counted in the report.

## Building

Requirements: Windows 10 or 11, Visual Studio 2022 Build Tools (C++
workload), CMake 3.25 or newer, and vcpkg. No GPU toolkit is needed.

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
build\tests\Release\neurograph_tests.exe
```

`vcpkg.json` pins the vcpkg release and the exact library versions. If the
packages are already installed under `build/vcpkg_installed`, the small
`vcpkg-toolchain.cmake` in this folder can be used instead of a full vcpkg
checkout.

To see the desktop app's screens without running it,
`neurograph_wizard_snapshots --csv data.csv --out shots` draws every
wizard step for that file, training included, to PNG files. It uses the
app's own drawing code, fonts and colours. `--size` and `--dpi` show
other window sizes and display scaling.

| CMake option | Default | Purpose |
|---|---|---|
| `NEUROGRAPH_BUILD_TESTS` | ON | Unit, integration and performance tests. |
| `NEUROGRAPH_ENABLE_CUDA` | OFF | Experimental GPU kernels and the legacy demo program; needs the CUDA toolkit. |
| `NEUROGRAPH_ENABLE_COVERAGE` | OFF | `neurograph_coverage` target (OpenCppCoverage). |
| `NEUROGRAPH_ENABLE_ASAN` | OFF | MSVC AddressSanitizer build. |

Logging goes to standard error at warning level. Set
`NEUROGRAPH_LOG_LEVEL=info` (or `debug`, `trace`) for more.

## Limitations

- **Windows only.** The desktop app uses Win32 and Direct3D 11, and the
  trial store uses DPAPI.
- **Tables only.** CSV files with one row per example. Images, audio,
  video and long free text are not supported.
- **Small files.** Under a few hundred rows, the 30% held back for checking
  costs a few points of accuracy compared with tools that use
  cross-validation, and the reported score rests on few rows (the report
  says so).
- **One CPU core.** Training is single-threaded. Files with tens of
  thousands of rows take under a minute; much larger files will be slow.
- **Licensing.** The licence check is built in, but this source tree has
  only a placeholder key, so builds from it run unlicensed with every
  feature available.

## The 0.1 engine

Version 0.1 trained a graph-based "self-organizing" network
(`src/core`, `src/training`). It was slow, ignored bias terms, and its
exported predictions were wrong. 0.2 replaces it in the automatic trainer
(`src/models`); the self-organizing idea lives on as the neural network
that grows its own hidden layer. The old engine still backs the
workbench's advanced panels, and model folders written by 0.1 still load.

## License

To be decided. All rights reserved until a `LICENSE` file is added.
