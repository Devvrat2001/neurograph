# Clean-VM Smoke Test

End-to-end rehearsal that proves the NeuroGraph release MSI installs
and runs cleanly on a vanilla Windows machine. Closes:

- **P1-07 AC line 627** — *"Runs on fresh Windows 10 VM with no Python,
  no Visual Studio."*
- **P1-10 AC line 840** — *"Tested on clean Windows 10 22H2 VM and
  Windows 11."*
- **P1-10 AC line 839** — *"Upgrades preserve license key + models"*
  (requires the v→v+1 path; do per release).
- **DoD E2E line** — *"Fresh Windows VM → install MSI → open wizard →
  titanic.csv → export .exe → run .exe on held-out rows → predictions
  match reference file."*

> Run before tagging any minor or major release. Major releases (1.x →
> 2.x) MUST run the full upgrade leg. Patch releases (1.0.x → 1.0.y)
> can skip the upgrade leg if no MSI version field changed. Expected
> wall time: ~30 min for the full pass on each VM image.

---

## VM provisioning

Two clean VM images are needed. **Both images must be vanilla** — no
Visual Studio, no Python, no any prior NeuroGraph install. Snapshot
each image so you can revert between rehearsals without re-provisioning.

- [ ] **Windows 10 22H2** image. Download free dev VHD from
      [Microsoft Edge dev tools](https://developer.microsoft.com/windows/downloads/virtual-machines/)
      or use any clean 22H2 ISO. Hyper-V / VirtualBox / VMware all fine.
- [ ] **Windows 11 (latest)** image. Same source.
- [ ] Each VM has at least 4 GB RAM, 30 GB disk, network access for
      the initial smoke (see Section 4 — license activation may need
      internet).
- [ ] Take a clean baseline snapshot named `pristine` on each VM.

---

## 1. Install (~5 min per VM)

For each VM image (Win10 + Win11):

1. [ ] Revert to `pristine` snapshot.
2. [ ] Copy `NeuroGraph-<version>.msi` from a release artifact to the
       VM (drag-drop, shared folder, or scp).
3. [ ] Right-click the MSI → **Install**. Confirm:
       - SmartScreen prompt names the publisher on the signing
         certificate (signed builds only — unsigned dev builds will warn
         "Unknown Publisher" and that's expected; note in sign-off).
       - Installer chooses default install dir `C:\Program Files\NeuroGraph`.
       - Finish completes without errors.
4. [ ] Verify install dir contents — at minimum:
       - `neurograph_workbench.exe`
       - `neurograph_predict.exe`
       - `neurograph_rt.dll`
       - `spdlog.dll`, `fmt.dll`, `libsodium.dll` (vcpkg deps)
5. [ ] Verify Start menu has a **NeuroGraph** entry.

---

## 2. Wizard end-to-end (~10 min per VM)

Drives the same path `tests/manual/wizard_smoke.md` covers, but on a
machine that has never seen the source tree. Catches missing-DLL,
missing-VCRUNTIME, registry-permission classes of bug that the
dev-machine wizard smoke can't.

1. [ ] Launch NeuroGraph from the Start menu. Window opens within ~3
       sec on cold start.
2. [ ] Wizard appears at **Step 1: Load Data** (not the legacy advanced
       UI — confirms `wizard_mode=true` shipped in the build).
3. [ ] Drag `datasets/iris.csv` (copy from the source repo to the VM
       desktop first) into the file picker. Preview populates.
4. [ ] Target column auto-picks `species`.
5. [ ] Click **Next** through Quality (no errors expected).
6. [ ] Click **Train** with Balanced budget. Progress bar advances,
       finishes within ~30 sec on the placeholder model. Wizard
       auto-advances to Results.
7. [ ] Hero card shows accuracy and the baseline verdict string.
8. [ ] Click **Export .exe**. Folder explorer opens at the bundle
       directory. Confirm `model.ngm`, `pipeline.ngpipe`,
       `manifest.json`, `card.md` all present.

---

## 3. Predict CLI on held-out rows (~5 min per VM)

The whole point of the C-track work — confirm the bundle the wizard
exports is consumable by `predict.exe` on a machine with no compiler.

1. [ ] Copy `datasets/iris_holdout.csv` to the VM (15 rows, no label
       column).
2. [ ] Open a Command Prompt (NOT PowerShell — exercise the lowest
       common denominator that customers will use).
3. [ ] Run:
       ```
       cd C:\Program Files\NeuroGraph
       neurograph_predict.exe --model "<exported_bundle_dir>" ^
                              --in     "<path>\iris_holdout.csv" ^
                              --out    "%TEMP%\preds.csv"
       ```
4. [ ] Exit code is 0 (`echo %ERRORLEVEL%`).
5. [ ] Output CSV has 16 lines (1 header + 15 predictions).
6. [ ] Header includes `predicted_class`, `p_setosa`, `p_versicolor`,
       `p_virginica` (exact column names depend on bundle's class
       labels).
7. [ ] **Reference diff:** copy a reference `preds.csv` from the
       previous release's smoke artifact. The first column
       (`predicted_class`) MUST match byte-for-byte. Probability
       columns may differ in the last 1-2 decimals across builds —
       file a bug only if argmax flips for any row.

---

## 4. License key flow (P1-11) — major releases only (~5 min per VM)

Skip on patch releases. Required on x.0.0 and x.y.0 to validate the
upgrade path preserves keys.

1. [ ] (If not first install on this image:) The wizard should detect
       the previously-stored license and skip the activation prompt.
2. [ ] (Fresh install:) Wizard prompts for license. Enter a known-good
       trial key from the engineering vault. Wizard accepts it within
       ~2 sec.
3. [ ] Close and re-launch. Activation should NOT prompt again — key
       persisted to `%APPDATA%\NeuroGraph\license.dat` (or wherever
       P1-11 settled on).

---

## 5. Upgrade leg (~10 min per VM, major releases only)

Validates P1-10 AC line 839 — *"installing v1.0.1 over v1.0.0 preserves
user license key (P1-11) and trained models."*

> Requires the **previous** release's MSI on disk. Fetch from
> Releases → assets before starting.

1. [ ] Revert to `pristine` snapshot.
2. [ ] Install the **previous** release's MSI (Section 1 steps 2-4).
3. [ ] Run Section 4 to seed a license, and run Section 2 to produce a
       saved bundle. Note the bundle path.
4. [ ] Without uninstalling, install the **current** release's MSI on
       top. Confirm the installer detects the prior version and runs
       a major upgrade (not a side-by-side install).
5. [ ] Re-launch the wizard. Verify:
       - Activation prompt does NOT appear (license preserved).
       - The bundle from step 3 still loads (Open existing → file
         picker → bundle dir → metrics + classes render correctly).
6. [ ] Check `Add or Remove Programs`: only **one** NeuroGraph entry
       at the new version (no orphaned old install).

---

## 6. Uninstall (~2 min per VM)

Last leg — verify clean removal so the next pristine-snapshot rehearsal
isn't required just to verify uninstall.

1. [ ] Settings → Apps → NeuroGraph → Uninstall.
2. [ ] No prompts, no errors. ~5 sec.
3. [ ] `C:\Program Files\NeuroGraph` is gone.
4. [ ] Start menu entry is gone.
5. [ ] User data NOT removed (license still under `%APPDATA%`,
       exported bundles still wherever the user saved them) — this is
       intentional; matches MSI convention.

---

## Sign-off

When every section's boxes are checked on **both VM images**, the
release is cleared for the VM-smoke half of the DoD.

- [ ] Date: ____________________
- [ ] Build SHA / tag: ____________________
- [ ] MSI version: ____________________
- [ ] Win10 22H2 reviewer: ____________________
- [ ] Win11 reviewer: ____________________
- [ ] Notes / deviations: ____________________

> Attach this completed checklist (signed off) to the release issue
> before publishing the GitHub Release.
