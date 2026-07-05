# Android On-Device Inference App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Kotlin + Jetpack Compose Android app that replays a CGM CSV stream through the LiteRT CompiledModel API on-device (float + INT8 models, float default), per `docs/superpowers/specs/2026-07-05-android-app-design.md`.

**Architecture:** Single-module Compose app at `android/` with package boundaries `replay/` (CSV loading, replay clock, sliding window), `inference/` (CompiledModel wrapper, quantization math, latency stats), `ui/` (one screen + ViewModel). Python scripts in `conversion/` produce the committed synthetic trace, the gitignored real trace, and golden vectors that an instrumented test uses to prove Kotlin↔Python parity.

**Tech Stack:** Kotlin 2.2.0, AGP 8.11.1, Gradle 8.13, Compose BOM 2025.11.00, Material 3, kotlinx-coroutines 1.10.2, `com.google.ai.edge.litert:litert:2.1.0` (CompiledModel API), JUnit 4, Python 3.12 tooling reusing `conversion/`.

## Global Constraints

- **No `INTERNET` permission, ever** — enforced by a Gradle `checkNoInternetPermission` task on the merged manifest, wired into `check`.
- **No GlucoBench data committed**: `real_trace.csv` stays gitignored; the committed trace and all golden vectors derive from synthetic data only.
- **No code, comment, or UI string may imply clinical validity or treatment guidance.** UI footer verbatim: `Portfolio demo on public research data — not a medical device, not treatment guidance.` Trend caption verbatim: `predicted trend, next 15 min`.
- INT8 input quantization is **round then clip to [-128, 127]** (`round(x/scale) + zeroPoint`), matching the corrected Python benchmark; scale/zero-point read from the model at runtime, never hardcoded.
- Hot inference path is **CompiledModel**; the classic `org.tensorflow.lite.Interpreter` (same artifact) may be used only for one-shot metadata reads at load time.
- Package `com.glucoedge.app`; applicationId `com.glucoedge.app`; minSdk 26, compileSdk 36, targetSdk 36.
- Version pins above are best-current-stable. If a pinned version fails to resolve, substitute the nearest stable, verify the build, and record the substitution in the commit message and task report (precedent: ai-edge-litert 2.1.5).
- Environment: `JAVA_HOME=/opt/homebrew/opt/openjdk@17`; Android SDK at `~/Library/Android/sdk`; AVD `Medium_Phone_API_35` available for instrumented tests. All `./gradlew` commands run from `android/`.
- Class order everywhere: `falling_fast, falling, stable, rising, rising_fast` (indices 0–4), matching `training/labeling.py`.
- Model input shape `(1, 1, 12)`, raw mg/dL floats; output 5 logits.

---

### Task 1: Gradle scaffold — empty Compose app that builds, with the no-INTERNET check

**Files:**
- Create: `android/settings.gradle.kts`, `android/build.gradle.kts`, `android/gradle.properties`, `android/gradle/libs.versions.toml`, `android/.gitignore`, `android/app/build.gradle.kts`, `android/app/src/main/AndroidManifest.xml`, `android/app/src/main/kotlin/com/glucoedge/app/MainActivity.kt`, `android/app/src/main/kotlin/com/glucoedge/app/ui/Theme.kt`, gradle wrapper files (`gradlew`, `gradlew.bat`, `gradle/wrapper/*`)
- Modify: none

**Interfaces:**
- Consumes: nothing.
- Produces: a building `:app` module all later Kotlin tasks compile inside; version catalog aliases (`libs.litert`, `libs.androidx.activity.compose`, etc.); the `checkNoInternetPermission` Gradle task.

- [ ] **Step 1: Install a gradle binary (wrapper bootstrap only) and generate the wrapper**

```bash
command -v gradle >/dev/null || brew install gradle
mkdir -p /Users/mohamednabil/Documents/vibe/GlucoEdge/android
cd /Users/mohamednabil/Documents/vibe/GlucoEdge/android
gradle wrapper --gradle-version 8.13
```

Expected: `gradlew`, `gradlew.bat`, `gradle/wrapper/gradle-wrapper.jar`, `gradle/wrapper/gradle-wrapper.properties` created.

- [ ] **Step 2: Write the Gradle project files**

`android/settings.gradle.kts`:
```kotlin
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "glucoedge-android"
include(":app")
```

`android/build.gradle.kts`:
```kotlin
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
```

`android/gradle.properties`:
```properties
org.gradle.jvmargs=-Xmx2g
android.useAndroidX=true
```

`android/gradle/libs.versions.toml`:
```toml
[versions]
agp = "8.11.1"
kotlin = "2.2.0"
composeBom = "2025.11.00"
activityCompose = "1.10.1"
lifecycle = "2.9.2"
coroutines = "1.10.2"
litert = "2.1.0"
junit = "4.13.2"

[libraries]
androidx-activity-compose = { module = "androidx.activity:activity-compose", version.ref = "activityCompose" }
androidx-lifecycle-viewmodel-compose = { module = "androidx.lifecycle:lifecycle-viewmodel-compose", version.ref = "lifecycle" }
compose-bom = { module = "androidx.compose:compose-bom", version.ref = "composeBom" }
compose-material3 = { module = "androidx.compose.material3:material3" }
compose-ui = { module = "androidx.compose.ui:ui" }
compose-ui-tooling-preview = { module = "androidx.compose.ui:ui-tooling-preview" }
coroutines-core = { module = "org.jetbrains.kotlinx:kotlinx-coroutines-core", version.ref = "coroutines" }
coroutines-test = { module = "org.jetbrains.kotlinx:kotlinx-coroutines-test", version.ref = "coroutines" }
litert = { module = "com.google.ai.edge.litert:litert", version.ref = "litert" }
junit = { module = "junit:junit", version.ref = "junit" }
androidx-test-runner = { module = "androidx.test:runner", version = "1.6.2" }
androidx-test-junit = { module = "androidx.test.ext:junit", version = "1.2.1" }

[plugins]
android-application = { id = "com.android.application", version.ref = "agp" }
kotlin-android = { id = "org.jetbrains.kotlin.android", version.ref = "kotlin" }
kotlin-compose = { id = "org.jetbrains.kotlin.plugin.compose", version.ref = "kotlin" }
```

`android/.gitignore`:
```
.gradle/
build/
local.properties
.idea/
*.iml
.DS_Store
```

`android/app/build.gradle.kts`:
```kotlin
import com.android.build.api.artifact.SingleArtifact

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.glucoedge.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.glucoedge.app"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
    sourceSets["main"].java.srcDirs("src/main/kotlin")
    sourceSets["test"].java.srcDirs("src/test/kotlin")
    sourceSets["androidTest"].java.srcDirs("src/androidTest/kotlin")
}

dependencies {
    implementation(platform(libs.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.compose.material3)
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.coroutines.core)
    implementation(libs.litert)
    testImplementation(libs.junit)
    testImplementation(libs.coroutines.test)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.junit)
}

// Structural no-network guarantee: fail `check` if the MERGED manifest
// (ours plus every dependency's) requests INTERNET.
androidComponents {
    onVariants(selector().withName("debug")) { variant ->
        val manifest = variant.artifacts.get(SingleArtifact.MERGED_MANIFEST)
        val checkTask = project.tasks.register("checkNoInternetPermission") {
            inputs.file(manifest)
            doLast {
                val text = manifest.get().asFile.readText()
                check(!text.contains("android.permission.INTERNET")) {
                    "Merged manifest must not request INTERNET permission (no network calls for inference, ever)"
                }
            }
        }
        project.tasks.named("check") { dependsOn(checkTask) }
    }
}
```

`android/app/src/main/AndroidManifest.xml` (note: **no** `<uses-permission>` at all):
```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application
        android:label="GlucoEdge"
        android:theme="@android:style/Theme.Material.Light.NoActionBar">
        <activity android:name=".MainActivity" android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

`android/app/src/main/kotlin/com/glucoedge/app/ui/Theme.kt`:
```kotlin
package com.glucoedge.app.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable

@Composable
fun GlucoEdgeTheme(content: @Composable () -> Unit) {
    MaterialTheme(content = content)
}
```

`android/app/src/main/kotlin/com/glucoedge/app/MainActivity.kt`:
```kotlin
package com.glucoedge.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.Text
import com.glucoedge.app.ui.GlucoEdgeTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { GlucoEdgeTheme { Text("GlucoEdge") } }
    }
}
```

- [ ] **Step 3: Create local.properties (gitignored) and build**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge/android
echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:assembleDebug :app:checkNoInternetPermission
```

Expected: `BUILD SUCCESSFUL`; the check task passes. If a version pin fails to resolve, substitute the nearest stable per Global Constraints and note it.

- [ ] **Step 4: Prove the check task actually fails when INTERNET appears**

Temporarily add `<uses-permission android:name="android.permission.INTERNET" />` to the manifest, run `./gradlew :app:checkNoInternetPermission` — Expected: **FAILS** with the guard message. Remove the line, rerun — Expected: passes. (This is the "run the test and watch it fail" step for a build-time guard.)

- [ ] **Step 5: Commit**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
git add android docs/superpowers/plans/2026-07-05-android-app.md
git commit -m "Scaffold Android Compose app with structural no-INTERNET guarantee"
```

---

### Task 2: Regenerate model artifacts, commit them as assets with provenance

**Files:**
- Create: `android/app/src/main/assets/trend_float.tflite`, `android/app/src/main/assets/trend_int8.tflite`, `android/app/src/main/assets/MODELS.md`
- Modify: `.gitignore` (scoped negations)

**Interfaces:**
- Consumes: `training/train.py` (`--seed 0 --save-checkpoint`), `conversion/convert.py`, `conversion/benchmark.py` (all on `main`).
- Produces: the two committed asset files (exact names above) that Tasks 4, 8, 9 load; `MODELS.md` sha256 lines in the format `` `<sha256>`  `trend_float.tflite` `` consumed by the parity test.

- [ ] **Step 1: Recreate the Python environment and GlucoBench link in the main checkout**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
ls GlucoBench || ln -s ../GlucoBench GlucoBench   # sibling clone, per CLAUDE.md; clone+unzip first if truly absent
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt -r conversion/requirements.txt
```

- [ ] **Step 2: Retrain the seeded checkpoint and convert**

```bash
venv/bin/python -m training.train --dataset weinstock --classes 5 --epochs 20 --seed 0 --save-checkpoint
venv/bin/python -m conversion.convert --dataset weinstock --classes 5
venv/bin/python -m conversion.benchmark --dataset weinstock --classes 5
```

Expected (verification against the documented record, `docs/superpowers/specs/2026-07-01-litert-conversion-results.md`): checkpoint accuracy 0.5184; float tflite ≈ 17,352 bytes, int8 ≈ 11,976 bytes; benchmark reports float acc 0.5184 / macro recall 0.5060 and int8 acc 0.5866 / macro recall 0.4135. If any number differs, STOP and report — do not commit unverified weights.

- [ ] **Step 3: Copy to assets, add scoped gitignore exceptions, write MODELS.md**

```bash
mkdir -p android/app/src/main/assets
cp results/model_float.tflite android/app/src/main/assets/trend_float.tflite
cp results/model_int8.tflite  android/app/src/main/assets/trend_int8.tflite
shasum -a 256 android/app/src/main/assets/*.tflite
```

Append to the repo root `.gitignore`, directly under the existing `*.tflite` line:
```
# Exception: the Android app's bundled model assets and synthetic trace are
# committed (our own trained weights / generated data, not GlucoBench data).
!android/app/src/main/assets/trend_float.tflite
!android/app/src/main/assets/trend_int8.tflite
!android/app/src/main/assets/synthetic_trace.csv
```

`android/app/src/main/assets/MODELS.md` (fill real values from the commands above):
```markdown
# Bundled model provenance

Both models were produced from this repository at commit <commit>, seed 0:
- `venv/bin/python -m training.train --dataset weinstock --classes 5 --epochs 20 --seed 0 --save-checkpoint`
- `venv/bin/python -m conversion.convert --dataset weinstock --classes 5`

Verified against docs/superpowers/specs/2026-07-01-litert-conversion-results.md
(float acc 0.5184 / macro recall 0.5060; int8 acc 0.5866 / macro recall 0.4135).

| file | sha256 | bytes |
|---|---|---|
| `trend_float.tflite` | `<sha256>` | <n> |
| `trend_int8.tflite` | `<sha256>` | <n> |

Not a medical device. Trained only on the public GlucoBench benchmark.
```

- [ ] **Step 4: Verify git sees exactly the intended files**

```bash
git status --short   # expect: .gitignore modified; 3 new files under android/app/src/main/assets/ (2 tflite + MODELS.md); nothing from results/
git check-ignore android/app/src/main/assets/trend_float.tflite && echo "STILL IGNORED - FIX" || echo OK
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore android/app/src/main/assets
git commit -m "Bundle float and INT8 models as app assets with provenance record"
```

---

### Task 3: Synthetic trace (committed) + real-trace extraction script (gitignored output)

**Files:**
- Create: `conversion/synthetic_trace.py`, `conversion/export_replay_trace.py`, `tests/test_synthetic_trace.py`, `android/app/src/main/assets/synthetic_trace.csv`
- Modify: none

**Interfaces:**
- Consumes: `training.labeling.label_trend(rate)` (existing), `training/train.py`'s `load_formatter(dataset)` (existing).
- Produces: CSV format both apps and scripts share — header `time,gl`, ISO-8601 minute timestamps, 5-minute cadence, glucose in mg/dL with 1 decimal. `synthetic_trace.csv` (288 rows, committed). `generate_trace() -> list[tuple[datetime, float]]` consumed by Task 4.

- [ ] **Step 1: Write the failing test**

`tests/test_synthetic_trace.py`:
```python
from datetime import timedelta

import numpy as np

from conversion.synthetic_trace import generate_trace
from training.labeling import label_trend


def test_trace_shape_and_cadence():
    trace = generate_trace()
    assert len(trace) == 288  # 24h at 5-min cadence
    deltas = {(b[0] - a[0]) for a, b in zip(trace, trace[1:])}
    assert deltas == {timedelta(minutes=5)}


def test_trace_values_plausible_and_cover_int8_ceiling():
    values = np.array([v for _, v in generate_trace()])
    assert values.min() >= 40.0
    assert values.max() <= 400.0
    assert values.max() > 240.0  # must exercise INT8 saturation region


def test_trace_produces_all_five_labels():
    values = [v for _, v in generate_trace()]
    labels = set()
    for i in range(len(values) - 15):
        rate = (values[i + 14] - values[i + 11]) / 15.0
        labels.add(label_trend(rate))
    assert labels == {"falling_fast", "falling", "stable", "rising", "rising_fast"}


def test_trace_is_deterministic():
    assert generate_trace() == generate_trace()
```

- [ ] **Step 2: Run it to make sure it fails**

```bash
venv/bin/python -m pytest tests/test_synthetic_trace.py -v
```
Expected: FAIL — `ModuleNotFoundError: conversion.synthetic_trace`.

- [ ] **Step 3: Implement the generator**

`conversion/synthetic_trace.py`:
```python
"""Deterministic synthetic CGM-like trace for the Android app demo.

Entirely generated data - no GlucoBench rows. Piecewise linear glucose
dynamics with small seeded noise, shaped so every trend class occurs and
the values exceed the INT8 model's ~240 mg/dL input ceiling.
"""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# (duration_minutes, slope mg/dL per minute)
_SEGMENTS = [
    (60, 0.0),    # steady baseline
    (45, 2.6),    # fast meal rise            -> rising_fast
    (30, 1.2),    # slowing rise              -> rising
    (40, 0.0),    # high plateau (> 240)
    (45, -2.6),   # fast correction drop      -> falling_fast
    (40, -1.2),   # slowing drop              -> falling
    (80, 0.0),    # steady
    (50, 2.2),    # second rise
    (60, -0.4),   # gentle drift down         -> stable-ish
    (90, -1.4),   # long fall
    (180, 0.05),  # overnight flat
    (300, 0.0),   # pad to 24h (trimmed below)
]
_START_MGDL = 110.0
_CADENCE_MIN = 5
_POINTS = 288  # 24h


def generate_trace() -> list[tuple[datetime, float]]:
    rng = np.random.default_rng(0)
    t = datetime(2026, 1, 1, 0, 0)
    value = _START_MGDL
    out = [(t, round(value, 1))]
    for duration, slope in _SEGMENTS:
        for _ in range(duration // _CADENCE_MIN):
            if len(out) >= _POINTS:
                break
            value = float(np.clip(value + slope * _CADENCE_MIN + rng.normal(0.0, 0.8), 45.0, 390.0))
            t += timedelta(minutes=_CADENCE_MIN)
            out.append((t, round(value, 1)))
    assert len(out) == _POINTS, f"segment table yields {len(out)} points, expected {_POINTS}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="android/app/src/main/assets/synthetic_trace.csv"
    )
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "gl"])
        for t, v in generate_trace():
            writer.writerow([t.strftime("%Y-%m-%dT%H:%M"), f"{v:.1f}"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

Note for the implementer: if `test_trace_produces_all_five_labels` or the ceiling test fails, adjust `_SEGMENTS` slopes/durations until all four tests pass — the tests are the contract, the segment table is tunable. Keep the total at exactly 288 points.

- [ ] **Step 4: Run the tests until they pass**

```bash
venv/bin/python -m pytest tests/test_synthetic_trace.py -v
```
Expected: 4 passed. Also run the full suite (`venv/bin/python -m pytest -q`) — no regressions.

- [ ] **Step 5: Generate and inspect the committed CSV**

```bash
venv/bin/python -m conversion.synthetic_trace
head -3 android/app/src/main/assets/synthetic_trace.csv
```
Expected: header `time,gl`, then rows like `2026-01-01T00:00,110.0`.

- [ ] **Step 6: Write the real-trace extraction script (output gitignored, run locally only)**

`conversion/export_replay_trace.py`:
```python
"""Extract one contiguous CGM segment from the local GlucoBench clone into
a replay CSV for the Android app. The output is GITIGNORED - GlucoBench
data is never committed to this repository."""
import argparse
import csv
from pathlib import Path

from training.train import load_formatter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="weinstock")
    parser.add_argument("--min-points", type=int, default=200)
    parser.add_argument(
        "--out", default="android/app/src/main/assets/real_trace.csv"
    )
    args = parser.parse_args()

    formatter = load_formatter(args.dataset)
    test = formatter.test_data
    # Longest single (id, id_segment) run with at least --min-points readings.
    groups = sorted(
        test.groupby(["id", "id_segment"]),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    key, seg = groups[0]
    if len(seg) < args.min_points:
        raise SystemExit(f"longest segment has only {len(seg)} points")
    seg = seg.sort_values("time")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "gl"])
        for _, row in seg.iterrows():
            writer.writerow(
                [row["time"].strftime("%Y-%m-%dT%H:%M"), f"{float(row['gl']):.1f}"]
            )
    print(f"wrote {path} ({len(seg)} readings from segment {key})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the extraction once and confirm it is ignored by git**

```bash
venv/bin/python -m conversion.export_replay_trace
git check-ignore android/app/src/main/assets/real_trace.csv && echo IGNORED-OK
```
Expected: `wrote ... (N readings ...)` with N ≥ 200, then `IGNORED-OK`. If `check-ignore` fails, STOP — do not commit; fix `.gitignore` first (the global `*.csv` rule should already cover it; the Task 2 negation must list `synthetic_trace.csv` only, never `real_trace.csv`).

- [ ] **Step 8: Commit (real_trace.csv must NOT appear in the diff)**

```bash
git add conversion/synthetic_trace.py conversion/export_replay_trace.py tests/test_synthetic_trace.py android/app/src/main/assets/synthetic_trace.csv
git status --short   # verify real_trace.csv is absent
git commit -m "Add synthetic replay trace (committed) and GlucoBench extraction script (output gitignored)"
```

---

### Task 4: Golden vectors for Kotlin↔Python parity

**Files:**
- Create: `conversion/export_golden_vectors.py`, `android/app/src/androidTest/assets/golden_vectors.json`
- Modify: none

**Interfaces:**
- Consumes: `conversion.benchmark`'s `_quantize_input(window, input_detail)` and `_dequantize_output(raw, output_detail)` (existing, corrected round+clip versions); `conversion.synthetic_trace.generate_trace()`; the two Task 2 asset files.
- Produces: `golden_vectors.json` with schema below, consumed verbatim by Task 8's instrumented test.

JSON schema (exact keys):
```json
{
  "class_names": ["falling_fast", "falling", "stable", "rising", "rising_fast"],
  "model_sha256": {"trend_float.tflite": "<hex>", "trend_int8.tflite": "<hex>"},
  "vectors": [
    {
      "window": [12 floats, raw mg/dL],
      "float_logits": [5 floats],
      "float_class": 0,
      "int8_raw_output": [5 ints, the raw int8 output values],
      "int8_class": 0
    }
  ]
}
```

- [ ] **Step 1: Write the export script**

`conversion/export_golden_vectors.py`:
```python
"""Export golden parity vectors for the Android instrumented test.

Windows come from the SYNTHETIC trace only - no GlucoBench data may enter
the repository via this file. Outputs are produced by the same corrected
quantize/dequantize path the benchmark uses, so the Android app is held to
exactly the numbers the conversion phase documented."""
import hashlib
import json
from pathlib import Path

import numpy as np

from conversion.benchmark import _dequantize_output, _quantize_input
from conversion.synthetic_trace import generate_trace

ASSETS = Path("android/app/src/main/assets")
OUT = Path("android/app/src/androidTest/assets/golden_vectors.json")
CLASS_NAMES = ["falling_fast", "falling", "stable", "rising", "rising_fast"]
WINDOW = 12
PER_CLASS_CAP = 4
MIN_TOTAL = 16


def _run(interpreter, window: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], _quantize_input(window, inp))
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])


def main() -> None:
    from ai_edge_litert.interpreter import Interpreter

    values = np.array([v for _, v in generate_trace()], dtype=np.float32)
    windows = [values[i : i + WINDOW] for i in range(len(values) - WINDOW + 1)]

    float_ip = Interpreter(model_path=str(ASSETS / "trend_float.tflite"))
    int8_ip = Interpreter(model_path=str(ASSETS / "trend_int8.tflite"))
    float_ip.allocate_tensors()
    int8_ip.allocate_tensors()
    int8_out_detail = int8_ip.get_output_details()[0]

    picked: dict[int, list[dict]] = {i: [] for i in range(5)}
    ceiling_window_included = False
    for w in windows:
        f_raw = _run(float_ip, w)[0]
        f_cls = int(np.argmax(f_raw))
        if len(picked[f_cls]) >= PER_CLASS_CAP and not (w.max() > 240 and not ceiling_window_included):
            continue
        q_raw = _run(int8_ip, w)[0]
        q_deq = _dequantize_output(np.array([q_raw]), int8_out_detail)[0]
        picked[f_cls].append(
            {
                "window": [round(float(x), 1) for x in w],
                "float_logits": [float(x) for x in f_raw],
                "float_class": f_cls,
                "int8_raw_output": [int(x) for x in q_raw],
                "int8_class": int(np.argmax(q_deq)),
            }
        )
        if w.max() > 240:
            ceiling_window_included = True

    vectors = [v for vs in picked.values() for v in vs]
    missing = [CLASS_NAMES[i] for i, vs in picked.items() if not vs]
    assert not missing, f"synthetic trace never float-predicts: {missing} - adjust _SEGMENTS"
    assert len(vectors) >= MIN_TOTAL, f"only {len(vectors)} vectors, need >= {MIN_TOTAL}"
    assert ceiling_window_included, "no window with values > 240 mg/dL made it in"

    sha = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ASSETS / "trend_float.tflite", ASSETS / "trend_int8.tflite"]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"class_names": CLASS_NAMES, "model_sha256": sha, "vectors": vectors},
            indent=1,
        )
    )
    print(f"wrote {OUT} ({len(vectors)} vectors, classes all covered)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (this is its own verification — the asserts are the test)**

```bash
venv/bin/python -m conversion.export_golden_vectors
venv/bin/python -c "
import json; d = json.load(open('android/app/src/androidTest/assets/golden_vectors.json'))
assert set(d) == {'class_names','model_sha256','vectors'} and len(d['vectors']) >= 16
print('classes:', sorted({v['float_class'] for v in d['vectors']}), 'count:', len(d['vectors']))"
```
Expected: `classes: [0, 1, 2, 3, 4]`, count ≥ 16. If a class is missing, tune `_SEGMENTS` in Task 3's generator (keeping its tests green) and regenerate BOTH the CSV and the goldens in the same commit.

- [ ] **Step 3: Cross-check sha256 against MODELS.md**

```bash
venv/bin/python -c "
import json, re
d = json.load(open('android/app/src/androidTest/assets/golden_vectors.json'))
md = open('android/app/src/main/assets/MODELS.md').read()
for name, sha in d['model_sha256'].items():
    assert sha in md, f'{name} sha mismatch vs MODELS.md'
print('sha256 consistent with MODELS.md')"
```
Expected: `sha256 consistent with MODELS.md`.

- [ ] **Step 4: Commit**

```bash
git add conversion/export_golden_vectors.py android/app/src/androidTest/assets/golden_vectors.json
git commit -m "Export golden parity vectors from the synthetic trace"
```

---

### Task 5: CsvTraceLoader + WindowBuffer (Kotlin, TDD)

**Files:**
- Create: `android/app/src/main/kotlin/com/glucoedge/app/replay/Reading.kt`, `android/app/src/main/kotlin/com/glucoedge/app/replay/CsvTraceLoader.kt`, `android/app/src/main/kotlin/com/glucoedge/app/replay/WindowBuffer.kt`
- Test: `android/app/src/test/kotlin/com/glucoedge/app/replay/CsvTraceLoaderTest.kt`, `android/app/src/test/kotlin/com/glucoedge/app/replay/WindowBufferTest.kt`

**Interfaces:**
- Consumes: nothing app-internal.
- Produces (exact signatures later tasks use):
```kotlin
data class Reading(val epochMinutes: Long, val mgdl: Float)
data class TraceLoadResult(val readings: List<Reading>, val skippedRows: Int)
object CsvTraceLoader { fun load(reader: java.io.Reader): TraceLoadResult }
class WindowBuffer(val windowSize: Int = 12, val cadenceMinutes: Long = 5) {
    data class AddResult(val window: FloatArray?, val wasReset: Boolean, val fill: Int)
    fun add(reading: Reading): AddResult
    fun reset()
}
```

- [ ] **Step 1: Write the failing loader tests**

`android/app/src/test/kotlin/com/glucoedge/app/replay/CsvTraceLoaderTest.kt`:
```kotlin
package com.glucoedge.app.replay

import java.io.StringReader
import org.junit.Assert.assertEquals
import org.junit.Test

class CsvTraceLoaderTest {
    private fun load(csv: String) = CsvTraceLoader.load(StringReader(csv))

    @Test fun parsesHeaderAndRows() {
        val r = load("time,gl\n2026-01-01T00:00,110.0\n2026-01-01T00:05,112.5\n")
        assertEquals(2, r.readings.size)
        assertEquals(0, r.skippedRows)
        assertEquals(110.0f, r.readings[0].mgdl, 1e-6f)
        assertEquals(5L, r.readings[1].epochMinutes - r.readings[0].epochMinutes)
    }

    @Test fun skipsAndCountsMalformedRows() {
        val r = load("time,gl\n2026-01-01T00:00,110.0\nnot-a-time,99\n2026-01-01T00:05,abc\n2026-01-01T00:10,120.0\n")
        assertEquals(2, r.readings.size)
        assertEquals(2, r.skippedRows)
    }

    @Test fun emptyBodyYieldsNoReadings() {
        val r = load("time,gl\n")
        assertEquals(0, r.readings.size)
    }
}
```

- [ ] **Step 2: Run and watch them fail**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge/android
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest --tests "com.glucoedge.app.replay.CsvTraceLoaderTest"
```
Expected: compilation FAILS (`Reading`/`CsvTraceLoader` unresolved).

- [ ] **Step 3: Implement Reading + loader**

`android/app/src/main/kotlin/com/glucoedge/app/replay/Reading.kt`:
```kotlin
package com.glucoedge.app.replay

/** One CGM reading. Time is epoch minutes so gap math is exact integer arithmetic. */
data class Reading(val epochMinutes: Long, val mgdl: Float)

data class TraceLoadResult(val readings: List<Reading>, val skippedRows: Int)
```

`android/app/src/main/kotlin/com/glucoedge/app/replay/CsvTraceLoader.kt`:
```kotlin
package com.glucoedge.app.replay

import java.io.Reader
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

/** Parses `time,gl` CSV (ISO-8601 minute timestamps, mg/dL values). */
object CsvTraceLoader {
    private val format = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm")

    fun load(reader: Reader): TraceLoadResult {
        val readings = mutableListOf<Reading>()
        var skipped = 0
        reader.buffered().useLines { lines ->
            lines.drop(1).forEach { line ->
                if (line.isBlank()) return@forEach
                val parts = line.split(',')
                val reading = runCatching {
                    val t = LocalDateTime.parse(parts[0].trim(), format)
                    Reading(t.toEpochSecond(ZoneOffset.UTC) / 60, parts[1].trim().toFloat())
                }.getOrNull()
                if (reading != null) readings.add(reading) else skipped++
            }
        }
        return TraceLoadResult(readings, skipped)
    }
}
```

- [ ] **Step 4: Loader tests pass**

Same command as Step 2. Expected: 3 pass.

- [ ] **Step 5: Write the failing WindowBuffer tests**

`android/app/src/test/kotlin/com/glucoedge/app/replay/WindowBufferTest.kt`:
```kotlin
package com.glucoedge.app.replay

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WindowBufferTest {
    private fun reading(i: Long, v: Float = 100f + i) = Reading(epochMinutes = i * 5, mgdl = v)

    @Test fun emitsNothingUntilTwelveReadings() {
        val buf = WindowBuffer()
        for (i in 0L..10L) {
            val r = buf.add(reading(i))
            assertNull(r.window)
            assertEquals((i + 1).toInt(), r.fill)
        }
        val full = buf.add(reading(11))
        assertEquals(12, full.window!!.size)
        assertEquals(100f, full.window!![0], 1e-6f)
        assertEquals(111f, full.window!![11], 1e-6f)
    }

    @Test fun slidesByOneAfterFull() {
        val buf = WindowBuffer()
        for (i in 0L..11L) buf.add(reading(i))
        val next = buf.add(reading(12))
        assertEquals(101f, next.window!![0], 1e-6f)
        assertEquals(112f, next.window!![11], 1e-6f)
    }

    @Test fun resetsOnTimeGap() {
        val buf = WindowBuffer()
        for (i in 0L..11L) buf.add(reading(i))
        // 25-minute jump: window must never span the gap (id_segment rule)
        val afterGap = buf.add(Reading(epochMinutes = 11 * 5 + 25, mgdl = 150f))
        assertTrue(afterGap.wasReset)
        assertNull(afterGap.window)
        assertEquals(1, afterGap.fill)
    }

    @Test fun exactCadenceIsNotAGap() {
        val buf = WindowBuffer()
        buf.add(reading(0))
        val r = buf.add(reading(1))
        assertFalse(r.wasReset)
    }
}
```

- [ ] **Step 6: Run and watch them fail, then implement**

`android/app/src/main/kotlin/com/glucoedge/app/replay/WindowBuffer.kt`:
```kotlin
package com.glucoedge.app.replay

/**
 * Sliding window of the last [windowSize] readings. Resets whenever the
 * time since the previous reading exceeds [cadenceMinutes] - the app-side
 * mirror of training's id_segment rule: a window must never span a gap in
 * the sensor record.
 */
class WindowBuffer(val windowSize: Int = 12, val cadenceMinutes: Long = 5) {
    data class AddResult(val window: FloatArray?, val wasReset: Boolean, val fill: Int)

    private val values = ArrayDeque<Float>()
    private var lastEpochMinutes: Long? = null

    fun add(reading: Reading): AddResult {
        val last = lastEpochMinutes
        val wasReset = last != null && (reading.epochMinutes - last) > cadenceMinutes
        if (wasReset) values.clear()
        lastEpochMinutes = reading.epochMinutes
        values.addLast(reading.mgdl)
        if (values.size > windowSize) values.removeFirst()
        val window = if (values.size == windowSize) values.toFloatArray() else null
        return AddResult(window, wasReset, values.size)
    }

    fun reset() {
        values.clear()
        lastEpochMinutes = null
    }
}
```

- [ ] **Step 7: All replay tests pass**

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest
```
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
git add android/app/src/main/kotlin/com/glucoedge/app/replay android/app/src/test
git commit -m "Add CSV trace loader and gap-aware sliding window buffer"
```

---

### Task 6: ReplayEngine (virtual-clock TDD)

**Files:**
- Create: `android/app/src/main/kotlin/com/glucoedge/app/replay/ReplayEngine.kt`
- Test: `android/app/src/test/kotlin/com/glucoedge/app/replay/ReplayEngineTest.kt`

**Interfaces:**
- Consumes: `Reading` (Task 5).
- Produces:
```kotlin
enum class Speed(val multiplier: Int) { X1(1), X4(4), X16(16) }
class ReplayEngine(private val readings: List<Reading>, baseIntervalMs: Long = 5_000L) {
    val events: kotlinx.coroutines.flow.Flow<Reading>  // cold; emits while playing, completes at end of trace
    fun play(); fun pause(); fun setSpeed(speed: Speed)
    val isPlaying: kotlinx.coroutines.flow.StateFlow<Boolean>
    val speed: kotlinx.coroutines.flow.StateFlow<Speed>
}
```
1× = one reading per `baseIntervalMs` (5 s of replay time per 5-min reading); 4× and 16× divide the interval.

- [ ] **Step 1: Write the failing tests**

`android/app/src/test/kotlin/com/glucoedge/app/replay/ReplayEngineTest.kt`:
```kotlin
package com.glucoedge.app.replay

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class ReplayEngineTest {
    private val trace = (0L..9L).map { Reading(it * 5, 100f + it) }

    @Test fun emitsNothingWhilePaused() = runTest {
        val engine = ReplayEngine(trace)
        val seen = mutableListOf<Reading>()
        val job = launch { engine.events.toList(seen) }
        advanceTimeBy(60_000); runCurrent()
        assertEquals(0, seen.size)
        job.cancel()
    }

    @Test fun emitsOnePerIntervalAtX1() = runTest {
        val engine = ReplayEngine(trace)
        val seen = mutableListOf<Reading>()
        val job = launch { engine.events.toList(seen) }
        engine.play()
        advanceTimeBy(15_000); runCurrent()   // 3 intervals of 5 s
        assertEquals(3, seen.size)
        assertEquals(trace[0], seen[0])
        job.cancel()
    }

    @Test fun speedChangeShortensInterval() = runTest {
        val engine = ReplayEngine(trace)
        val seen = mutableListOf<Reading>()
        val job = launch { engine.events.toList(seen) }
        engine.setSpeed(Speed.X16)            // 5000/16 = 312.5 -> 312 ms
        engine.play()
        advanceTimeBy(3_120); runCurrent()    // 10 intervals: whole trace
        assertEquals(10, seen.size)
        job.cancel()
    }

    @Test fun pauseStopsEmission() = runTest {
        val engine = ReplayEngine(trace)
        val seen = mutableListOf<Reading>()
        val job = launch { engine.events.toList(seen) }
        engine.play()
        advanceTimeBy(10_000); runCurrent()
        engine.pause()
        advanceTimeBy(60_000); runCurrent()
        assertEquals(2, seen.size)
        job.cancel()
    }
}
```

- [ ] **Step 2: Run and watch them fail**

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest --tests "com.glucoedge.app.replay.ReplayEngineTest"
```
Expected: compilation FAILS (`ReplayEngine`/`Speed` unresolved).

- [ ] **Step 3: Implement**

`android/app/src/main/kotlin/com/glucoedge/app/replay/ReplayEngine.kt`:
```kotlin
package com.glucoedge.app.replay

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow

enum class Speed(val multiplier: Int) { X1(1), X4(4), X16(16) }

/**
 * Replays a fixed trace on a virtual clock: one reading per interval while
 * playing. Decoupled from wall time - the real 5-minute CGM cadence is
 * compressed to [baseIntervalMs] per reading at 1x.
 */
class ReplayEngine(
    private val readings: List<Reading>,
    private val baseIntervalMs: Long = 5_000L,
) {
    private val _isPlaying = MutableStateFlow(false)
    private val _speed = MutableStateFlow(Speed.X1)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()
    val speed: StateFlow<Speed> = _speed.asStateFlow()

    val events: Flow<Reading> = flow {
        for (reading in readings) {
            _isPlaying.first { it }                       // suspend until playing
            delay(baseIntervalMs / _speed.value.multiplier)
            emit(reading)
        }
    }

    fun play() { _isPlaying.value = true }
    fun pause() { _isPlaying.value = false }
    fun setSpeed(speed: Speed) { _speed.value = speed }
}
```

- [ ] **Step 4: Tests pass**

Same command as Step 2. Expected: 4 pass. (If the pause test is flaky because a `delay` was already in flight when `pause()` landed, it may deliver one extra reading — if observed, fix the engine by re-checking `_isPlaying` after the delay: `_isPlaying.first { it }; delay(...); _isPlaying.first { it }; emit(...)` — and keep the test's expected count.)

- [ ] **Step 5: Commit**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
git add android/app/src/main/kotlin/com/glucoedge/app/replay/ReplayEngine.kt android/app/src/test/kotlin/com/glucoedge/app/replay/ReplayEngineTest.kt
git commit -m "Add virtual-clock replay engine with play/pause/speed"
```

---

### Task 7: Quantization math + LatencyMeter (pure Kotlin, TDD)

**Files:**
- Create: `android/app/src/main/kotlin/com/glucoedge/app/inference/QuantizationMath.kt`, `android/app/src/main/kotlin/com/glucoedge/app/inference/LatencyMeter.kt`
- Test: `android/app/src/test/kotlin/com/glucoedge/app/inference/QuantizationMathTest.kt`, `android/app/src/test/kotlin/com/glucoedge/app/inference/LatencyMeterTest.kt`

**Interfaces:**
- Consumes: nothing app-internal.
- Produces:
```kotlin
object QuantizationMath {
    fun quantizeInt8(values: FloatArray, scale: Float, zeroPoint: Int): ByteArray
    fun dequantizeInt8(values: ByteArray, scale: Float, zeroPoint: Int): FloatArray
    fun softmax(logits: FloatArray): FloatArray
}
data class LatencyStats(val count: Int, val meanMs: Double, val medianMs: Double, val p95Ms: Double)
class LatencyMeter(val capacity: Int = 100) { fun record(nanos: Long); fun reset(); val stats: LatencyStats }
```

- [ ] **Step 1: Write the failing quantization tests** (these encode the exact bug class the conversion-phase review caught — the clip is load-bearing)

`android/app/src/test/kotlin/com/glucoedge/app/inference/QuantizationMathTest.kt`:
```kotlin
package com.glucoedge.app.inference

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class QuantizationMathTest {
    // The INT8 model's real params from the conversion record:
    private val scale = 0.9408126473426819f
    private val zeroPoint = -128

    @Test fun quantizesInRangeValueByRoundThenOffset() {
        // 110 / 0.9408... = 116.92 -> round 117; 117 + (-128) = -11
        val q = QuantizationMath.quantizeInt8(floatArrayOf(110f), scale, zeroPoint)
        assertEquals((-11).toByte(), q[0])
    }

    @Test fun clipsAboveCeilingInsteadOfWrapping() {
        // 260 mg/dL is above the ~239.9 ceiling: must clip to +127, never wrap negative
        val q = QuantizationMath.quantizeInt8(floatArrayOf(260f, 401f), scale, zeroPoint)
        assertArrayEquals(byteArrayOf(127, 127), q)
    }

    @Test fun clipsBelowFloor() {
        val q = QuantizationMath.quantizeInt8(floatArrayOf(-50f), scale, zeroPoint)
        assertEquals((-128).toByte(), q[0])
    }

    @Test fun dequantizeRoundTripsWithinScale() {
        val original = floatArrayOf(80f, 110f, 180f, 239f)
        val q = QuantizationMath.quantizeInt8(original, scale, zeroPoint)
        val back = QuantizationMath.dequantizeInt8(q, scale, zeroPoint)
        original.zip(back.toTypedArray()).forEach { (a, b) ->
            assertEquals(a, b, scale)  // error bounded by one quantization step
        }
    }

    @Test fun softmaxIsStableAndNormalized() {
        val p = QuantizationMath.softmax(floatArrayOf(1000f, 1001f, 999f, 0f, -1000f))
        assertEquals(1.0f, p.sum(), 1e-5f)
        assertEquals(4, p.indexOfFirst { it == p.max() }.let { if (p[1] > p[0]) 1 else it })
    }
}
```

(Fix the last assertion while implementing if it reads awkwardly — its intent: probabilities sum to 1 and the max sits at index 1; large logits must not overflow to NaN.)

- [ ] **Step 2: Run, watch fail, implement**

`android/app/src/main/kotlin/com/glucoedge/app/inference/QuantizationMath.kt`:
```kotlin
package com.glucoedge.app.inference

import kotlin.math.exp
import kotlin.math.roundToInt

object QuantizationMath {
    /**
     * Affine int8 quantization: round(x/scale) + zeroPoint, CLIPPED to
     * [-128, 127]. The clip is load-bearing: without it, glucose readings
     * above the representable ceiling wrap around int8 (260 mg/dL would
     * become ~19 mg/dL) - the exact bug the conversion-phase review caught
     * in the Python benchmark.
     */
    fun quantizeInt8(values: FloatArray, scale: Float, zeroPoint: Int): ByteArray =
        ByteArray(values.size) { i ->
            val q = (values[i] / scale).roundToInt() + zeroPoint
            q.coerceIn(-128, 127).toByte()
        }

    fun dequantizeInt8(values: ByteArray, scale: Float, zeroPoint: Int): FloatArray =
        FloatArray(values.size) { i -> (values[i].toInt() - zeroPoint) * scale }

    fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }
}
```

Run: `JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest --tests "com.glucoedge.app.inference.QuantizationMathTest"` — Expected: 5 pass.

- [ ] **Step 3: Write the failing LatencyMeter tests, then implement**

`android/app/src/test/kotlin/com/glucoedge/app/inference/LatencyMeterTest.kt`:
```kotlin
package com.glucoedge.app.inference

import org.junit.Assert.assertEquals
import org.junit.Test

class LatencyMeterTest {
    @Test fun statsOverKnownValues() {
        val m = LatencyMeter()
        // 1..100 ms recorded as nanos
        (1..100).forEach { m.record(it * 1_000_000L) }
        val s = m.stats
        assertEquals(100, s.count)
        assertEquals(50.5, s.meanMs, 1e-9)
        assertEquals(50.5, s.medianMs, 1e-9)
        assertEquals(95.0, s.p95Ms, 1.0)
    }

    @Test fun rollsOverCapacity() {
        val m = LatencyMeter(capacity = 10)
        (1..20).forEach { m.record(it * 1_000_000L) }
        assertEquals(10, m.stats.count)
        assertEquals(15.5, m.stats.meanMs, 1e-9)  // only 11..20 retained
    }

    @Test fun emptyMeterIsAllZero() {
        val s = LatencyMeter().stats
        assertEquals(0, s.count)
        assertEquals(0.0, s.meanMs, 0.0)
    }

    @Test fun resetClears() {
        val m = LatencyMeter()
        m.record(5_000_000L); m.reset()
        assertEquals(0, m.stats.count)
    }
}
```

`android/app/src/main/kotlin/com/glucoedge/app/inference/LatencyMeter.kt`:
```kotlin
package com.glucoedge.app.inference

data class LatencyStats(val count: Int, val meanMs: Double, val medianMs: Double, val p95Ms: Double)

/** Rolling window of the last [capacity] inference wall times. */
class LatencyMeter(val capacity: Int = 100) {
    private val samplesNanos = ArrayDeque<Long>()

    @Synchronized
    fun record(nanos: Long) {
        samplesNanos.addLast(nanos)
        if (samplesNanos.size > capacity) samplesNanos.removeFirst()
    }

    @Synchronized
    fun reset() = samplesNanos.clear()

    val stats: LatencyStats
        @Synchronized get() {
            if (samplesNanos.isEmpty()) return LatencyStats(0, 0.0, 0.0, 0.0)
            val ms = samplesNanos.map { it / 1e6 }.sorted()
            val median = if (ms.size % 2 == 1) ms[ms.size / 2]
                         else (ms[ms.size / 2 - 1] + ms[ms.size / 2]) / 2.0
            val p95 = ms[((ms.size - 1) * 95) / 100]
            return LatencyStats(ms.size, ms.average(), median, p95)
        }
}
```

Run: `JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest` — Expected: all pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
git add android/app/src/main/kotlin/com/glucoedge/app/inference android/app/src/test/kotlin/com/glucoedge/app/inference
git commit -m "Add round-and-clip INT8 quantization math and rolling latency meter"
```

---

### Task 8: TrendClassifier (CompiledModel) + golden-vector parity instrumented test

**Files:**
- Create: `android/app/src/main/kotlin/com/glucoedge/app/inference/TrendClassifier.kt`
- Test: `android/app/src/androidTest/kotlin/com/glucoedge/app/GoldenParityTest.kt`

**Interfaces:**
- Consumes: `QuantizationMath`, `LatencyMeter` (Task 7); assets from Tasks 2 & 4.
- Produces:
```kotlin
enum class ModelFile(val assetName: String) {
    FLOAT("trend_float.tflite"), INT8("trend_int8.tflite")
}
data class Prediction(val probabilities: FloatArray, val classIndex: Int, val latencyNanos: Long)
interface Classifier { fun classify(window: FloatArray): Prediction; fun close() }
class TrendClassifier private constructor(...) : Classifier {
    companion object {
        val CLASS_NAMES = listOf("falling_fast", "falling", "stable", "rising", "rising_fast")
        fun create(context: android.content.Context, model: ModelFile): TrendClassifier
    }
}
```

- [ ] **Step 1: Implement TrendClassifier**

`android/app/src/main/kotlin/com/glucoedge/app/inference/TrendClassifier.kt`:
```kotlin
package com.glucoedge.app.inference

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

enum class ModelFile(val assetName: String) {
    FLOAT("trend_float.tflite"), INT8("trend_int8.tflite")
}

data class Prediction(val probabilities: FloatArray, val classIndex: Int, val latencyNanos: Long)

interface Classifier {
    fun classify(window: FloatArray): Prediction
    fun close()
}

private data class QuantParams(val scale: Float, val zeroPoint: Int)

/**
 * Wraps a LiteRT CompiledModel (the hot path). For the INT8 model, input
 * quantization params are read ONCE at load time via the classic
 * Interpreter API (same artifact) - never hardcoded - and the Kotlin
 * quantization replicates the corrected Python benchmark: round then clip.
 */
class TrendClassifier private constructor(
    private val compiledModel: CompiledModel,
    private val inputQuant: QuantParams?,   // null => float model
    private val outputQuant: QuantParams?,
) : Classifier {
    private val inputBuffers = compiledModel.createInputBuffers()
    private val outputBuffers = compiledModel.createOutputBuffers()

    override fun classify(window: FloatArray): Prediction {
        require(window.size == 12) { "expected a 12-reading window" }
        val start = System.nanoTime()
        if (inputQuant == null) {
            inputBuffers[0].writeFloat(window)
        } else {
            inputBuffers[0].writeInt8(
                QuantizationMath.quantizeInt8(window, inputQuant.scale, inputQuant.zeroPoint)
            )
        }
        compiledModel.run(inputBuffers, outputBuffers)
        val logits = if (outputQuant == null) {
            outputBuffers[0].readFloat()
        } else {
            QuantizationMath.dequantizeInt8(
                outputBuffers[0].readInt8(), outputQuant.scale, outputQuant.zeroPoint
            )
        }
        val elapsed = System.nanoTime() - start
        val probs = QuantizationMath.softmax(logits)
        val cls = probs.indices.maxBy { probs[it] }
        return Prediction(probs, cls, elapsed)
    }

    override fun close() {
        compiledModel.close()
    }

    companion object {
        val CLASS_NAMES = listOf("falling_fast", "falling", "stable", "rising", "rising_fast")

        fun create(context: Context, model: ModelFile): TrendClassifier {
            val file = File(context.cacheDir, model.assetName)
            context.assets.open(model.assetName).use { input ->
                file.outputStream().use { input.copyTo(it) }
            }
            val (inQ, outQ) = readQuantParams(file, quantized = model == ModelFile.INT8)
            val compiled = CompiledModel.create(
                file.absolutePath, CompiledModel.Options(Accelerator.CPU)
            )
            return TrendClassifier(compiled, inQ, outQ)
        }

        /** One-shot metadata read via the classic Interpreter (same litert artifact). */
        private fun readQuantParams(file: File, quantized: Boolean): Pair<QuantParams?, QuantParams?> {
            if (!quantized) return null to null
            val buffer: ByteBuffer = file.readBytes().let {
                ByteBuffer.allocateDirect(it.size).order(ByteOrder.nativeOrder()).put(it).apply { rewind() }
            }
            org.tensorflow.lite.Interpreter(buffer).use { interpreter ->
                val inQ = interpreter.getInputTensor(0).quantizationParams()
                val outQ = interpreter.getOutputTensor(0).quantizationParams()
                return QuantParams(inQ.scale, inQ.zeroPoint) to QuantParams(outQ.scale, outQ.zeroPoint)
            }
        }
    }
}
```

**API-surface decision ladder** (resolve while making this compile; document which rung was used in the task report):
1. `TensorBuffer.writeInt8(ByteArray)` / `readInt8()` as written above.
2. If those methods don't exist in litert 2.1.0, look for `write(ByteArray)` / `read...` byte variants on the buffer class and use them.
3. If the CompiledModel TensorBuffer genuinely has no int8 write path, run the INT8 model through `org.tensorflow.lite.Interpreter` instead (float stays on CompiledModel) and record the deviation prominently in the task report and `MODELS.md` — the parity test still gates correctness.
Similarly, if `Interpreter.use {}` doesn't compile (not `AutoCloseable`), call `.close()` in a `finally`.

- [ ] **Step 2: Compile**

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:assembleDebug
```
Expected: BUILD SUCCESSFUL (adjusting per the ladder above if needed).

- [ ] **Step 3: Write the golden parity instrumented test**

`android/app/src/androidTest/kotlin/com/glucoedge/app/GoldenParityTest.kt`:
```kotlin
package com.glucoedge.app

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.glucoedge.app.inference.ModelFile
import com.glucoedge.app.inference.TrendClassifier
import java.security.MessageDigest
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GoldenParityTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val testContext = InstrumentationRegistry.getInstrumentation().context

    private fun goldens(): JSONObject =
        JSONObject(testContext.assets.open("golden_vectors.json").bufferedReader().readText())

    private fun sha256OfAsset(name: String): String {
        val bytes = appContext.assets.open(name).readBytes()
        return MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
    }

    @Test
    fun bundledModelsMatchGoldenHashes() {
        val expected = goldens().getJSONObject("model_sha256")
        for (model in ModelFile.entries) {
            assertEquals(
                "asset ${model.assetName} drifted from the goldens - regenerate both together",
                expected.getString(model.assetName),
                sha256OfAsset(model.assetName),
            )
        }
    }

    @Test
    fun floatModelMatchesPythonLogitsAndClasses() {
        val vectors = goldens().getJSONArray("vectors")
        val classifier = TrendClassifier.create(appContext, ModelFile.FLOAT)
        try {
            for (i in 0 until vectors.length()) {
                val v = vectors.getJSONObject(i)
                val window = v.getJSONArray("window").let { arr ->
                    FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
                }
                val prediction = classifier.classify(window)
                assertEquals("vector $i class", v.getInt("float_class"), prediction.classIndex)
            }
        } finally {
            classifier.close()
        }
    }

    @Test
    fun int8ModelMatchesPythonClasses() {
        val vectors = goldens().getJSONArray("vectors")
        val classifier = TrendClassifier.create(appContext, ModelFile.INT8)
        try {
            for (i in 0 until vectors.length()) {
                val v = vectors.getJSONObject(i)
                val window = v.getJSONArray("window").let { arr ->
                    FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
                }
                val prediction = classifier.classify(window)
                assertEquals("vector $i int8 class", v.getInt("int8_class"), prediction.classIndex)
            }
        } finally {
            classifier.close()
        }
    }
}
```

(Class-level parity is the required gate. If the buffer API rung 1 worked, ALSO strengthen the float test by asserting each logit within 1e-4 of `float_logits` — softmax-then-argmax can mask logit drift. If a looser rung was needed, class parity plus a written note is acceptable.)

- [ ] **Step 4: Boot the emulator and run the parity test**

```bash
~/Library/Android/sdk/emulator/emulator -avd Medium_Phone_API_35 -no-window -no-audio -no-boot-anim &
~/Library/Android/sdk/platform-tools/adb wait-for-device
until [ "$(~/Library/Android/sdk/platform-tools/adb shell getprop sys.boot_completed | tr -d '\r')" = "1" ]; do sleep 3; done
cd /Users/mohamednabil/Documents/vibe/GlucoEdge/android
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:connectedDebugAndroidTest
```
Expected: 3 tests pass. This is the correctness anchor for the whole app — if class parity fails, debug the Kotlin input layout/quantization (compare a single window's quantized bytes against `_quantize_input` output in Python) rather than loosening the assertion. Leave the emulator running for Task 9.

- [ ] **Step 5: Commit**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
git add android/app/src/main/kotlin/com/glucoedge/app/inference/TrendClassifier.kt android/app/src/androidTest/kotlin
git commit -m "Add CompiledModel classifier with golden-vector parity proven on-device"
```

---

### Task 9: ViewModel + Compose UI

**Files:**
- Create: `android/app/src/main/kotlin/com/glucoedge/app/ui/MainViewModel.kt`, `android/app/src/main/kotlin/com/glucoedge/app/ui/MainScreen.kt`
- Modify: `android/app/src/main/kotlin/com/glucoedge/app/MainActivity.kt`
- Test: `android/app/src/test/kotlin/com/glucoedge/app/ui/MainViewModelTest.kt`

**Interfaces:**
- Consumes: everything from Tasks 5–8 (exact signatures in those tasks).
- Produces: `MainViewModel(traceSource: TraceSource, classifierFactory: (ModelFile) -> Classifier)`, `data class UiState(...)` below, `MainScreen(viewModel)` composable.

- [ ] **Step 1: Write the failing ViewModel tests (fakes, virtual clock)**

`android/app/src/test/kotlin/com/glucoedge/app/ui/MainViewModelTest.kt`:
```kotlin
package com.glucoedge.app.ui

import com.glucoedge.app.inference.Classifier
import com.glucoedge.app.inference.Prediction
import com.glucoedge.app.replay.Reading
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

private class FakeClassifier : Classifier {
    var calls = 0
    override fun classify(window: FloatArray): Prediction {
        calls++
        return Prediction(floatArrayOf(0.1f, 0.1f, 0.6f, 0.1f, 0.1f), 2, 1_000_000L)
    }
    override fun close() {}
}

class MainViewModelTest {
    private val dispatcher = StandardTestDispatcher()
    private val trace = (0L..19L).map { Reading(it * 5, 100f + it) }
    private lateinit var fake: FakeClassifier

    @Before fun setUp() {
        Dispatchers.setMain(dispatcher)
        fake = FakeClassifier()
    }
    @After fun tearDown() = Dispatchers.resetMain()

    private fun viewModel() = MainViewModel(
        traceSource = TraceSource(trace, skippedRows = 1, label = "synthetic"),
        classifierFactory = { fake },
    )

    @Test fun noPredictionUntilWindowFills() = runTest(dispatcher) {
        val vm = viewModel()
        vm.onPlayPause()                    // start playing
        advanceTimeBy(11 * 5_000L); runCurrent()
        assertNull(vm.uiState.value.prediction)
        assertEquals(0, fake.calls)
        advanceTimeBy(5_000L); runCurrent() // 12th reading
        assertNotNull(vm.uiState.value.prediction)
        assertEquals(2, vm.uiState.value.prediction!!.classIndex)
        assertEquals(1, fake.calls)
    }

    @Test fun exposesTraceMetadata() = runTest(dispatcher) {
        val vm = viewModel()
        assertEquals("synthetic", vm.uiState.value.traceLabel)
        assertEquals(1, vm.uiState.value.skippedRows)
    }

    @Test fun modelToggleSwapsClassifierAndResetsStats() = runTest(dispatcher) {
        val vm = viewModel()
        vm.onPlayPause()
        advanceTimeBy(12 * 5_000L); runCurrent()
        assertEquals(1, fake.calls)
        vm.onToggleModel()
        runCurrent()
        assertEquals(com.glucoedge.app.inference.ModelFile.INT8, vm.uiState.value.model)
        assertEquals(0, vm.uiState.value.stats.count)
    }
}
```

- [ ] **Step 2: Run, watch fail (unresolved `MainViewModel`/`TraceSource`/`UiState`)**

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest --tests "com.glucoedge.app.ui.MainViewModelTest"
```

- [ ] **Step 3: Implement the ViewModel**

`android/app/src/main/kotlin/com/glucoedge/app/ui/MainViewModel.kt`:
```kotlin
package com.glucoedge.app.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.glucoedge.app.inference.Classifier
import com.glucoedge.app.inference.LatencyMeter
import com.glucoedge.app.inference.LatencyStats
import com.glucoedge.app.inference.ModelFile
import com.glucoedge.app.inference.Prediction
import com.glucoedge.app.replay.Reading
import com.glucoedge.app.replay.ReplayEngine
import com.glucoedge.app.replay.Speed
import com.glucoedge.app.replay.WindowBuffer
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class TraceSource(val readings: List<Reading>, val skippedRows: Int, val label: String)

data class UiState(
    val recentReadings: List<Float> = emptyList(),   // up to last 12 mgdl values
    val currentMgdl: Float? = null,
    val prediction: Prediction? = null,
    val model: ModelFile = ModelFile.FLOAT,
    val playing: Boolean = false,
    val speed: Speed = Speed.X1,
    val stats: LatencyStats = LatencyStats(0, 0.0, 0.0, 0.0),
    val skippedRows: Int = 0,
    val rebuffering: Boolean = false,
    val traceLabel: String = "",
    val error: String? = null,
)

class MainViewModel(
    private val traceSource: TraceSource,
    private val classifierFactory: (ModelFile) -> Classifier,
) : ViewModel() {
    private val _uiState = MutableStateFlow(
        UiState(skippedRows = traceSource.skippedRows, traceLabel = traceSource.label)
    )
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private val engine = ReplayEngine(traceSource.readings)
    private val buffer = WindowBuffer()
    private val meter = LatencyMeter()
    private var classifier: Classifier? = null

    init {
        if (traceSource.readings.isEmpty()) {
            _uiState.update { it.copy(error = "No valid readings in trace") }
        } else {
            classifier = classifierFactory(ModelFile.FLOAT)
            viewModelScope.launch {
                engine.events.collect { reading -> onReading(reading) }
            }
        }
    }

    private fun onReading(reading: Reading) {
        val result = buffer.add(reading)
        val prediction = result.window?.let { window ->
            runCatching { classifier?.classify(window) }
                .onFailure { e -> _uiState.update { s -> s.copy(error = "Inference failed: ${e.message}") } }
                .getOrNull()
        }
        prediction?.let { meter.record(it.latencyNanos) }
        _uiState.update { s ->
            s.copy(
                recentReadings = (s.recentReadings + reading.mgdl).takeLast(12),
                currentMgdl = reading.mgdl,
                prediction = prediction ?: if (result.wasReset) null else s.prediction,
                rebuffering = result.window == null,
                stats = meter.stats,
            )
        }
    }

    fun onPlayPause() {
        if (engine.isPlaying.value) engine.pause() else engine.play()
        _uiState.update { it.copy(playing = engine.isPlaying.value) }
    }

    fun onSpeed(speed: Speed) {
        engine.setSpeed(speed)
        _uiState.update { it.copy(speed = speed) }
    }

    fun onToggleModel() {
        val next = if (_uiState.value.model == ModelFile.FLOAT) ModelFile.INT8 else ModelFile.FLOAT
        classifier?.close()
        classifier = classifierFactory(next)
        meter.reset()
        _uiState.update { it.copy(model = next, stats = meter.stats, prediction = null) }
    }

    override fun onCleared() {
        classifier?.close()
    }
}
```

- [ ] **Step 4: ViewModel tests pass**

Same command as Step 2. Expected: 3 pass.

- [ ] **Step 5: Implement the screen and wire MainActivity**

`android/app/src/main/kotlin/com/glucoedge/app/ui/MainScreen.kt`:
```kotlin
package com.glucoedge.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.glucoedge.app.inference.TrendClassifier
import com.glucoedge.app.replay.Speed

private val TREND_ARROWS = listOf("↓↓", "↓", "→", "↑", "↑↑")

@Composable
fun MainScreen(viewModel: MainViewModel, deviceLabel: String) {
    val state by viewModel.uiState.collectAsState()

    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        state.error?.let { Text(it, color = MaterialTheme.colorScheme.error); return@Column }

        GlucoseChart(state.recentReadings)

        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("${state.currentMgdl?.let { "%.0f".format(it) } ?: "--"} mg/dL",
                 style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.width(8.dp))
            Text("(${state.traceLabel} trace)", style = MaterialTheme.typography.bodySmall)
        }

        if (state.rebuffering) Text("window rebuffering…", style = MaterialTheme.typography.bodySmall)

        state.prediction?.let { p ->
            Text("${TREND_ARROWS[p.classIndex]}  ${TrendClassifier.CLASS_NAMES[p.classIndex]}",
                 style = MaterialTheme.typography.headlineLarge)
            Text("predicted trend, next 15 min", style = MaterialTheme.typography.bodySmall)
            TrendClassifier.CLASS_NAMES.forEachIndexed { i, name ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(name, Modifier.width(96.dp), style = MaterialTheme.typography.bodySmall)
                    LinearProgressIndicator(
                        progress = { p.probabilities[i] },
                        modifier = Modifier.fillMaxWidth().height(8.dp),
                    )
                }
            }
        }

        Text(
            "model: ${state.model.name.lowercase()}  ·  inferences: ${state.stats.count}  ·  " +
            "mean %.3f ms  ·  p95 %.3f ms  ·  skipped rows: %d  ·  %s"
                .format(state.stats.meanMs, state.stats.p95Ms, state.skippedRows, deviceLabel),
            style = MaterialTheme.typography.bodySmall,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::onPlayPause) { Text(if (state.playing) "Pause" else "Play") }
            Speed.entries.forEach { s ->
                OutlinedButton(onClick = { viewModel.onSpeed(s) }) {
                    Text(if (s == state.speed) "[${s.multiplier}×]" else "${s.multiplier}×")
                }
            }
            OutlinedButton(onClick = viewModel::onToggleModel) {
                Text(if (state.model.name == "FLOAT") "→ INT8" else "→ float")
            }
        }

        Spacer(Modifier.weight(1f))
        Text(
            "Portfolio demo on public research data — not a medical device, not treatment guidance.",
            style = MaterialTheme.typography.labelSmall,
        )
    }
}

@Composable
private fun GlucoseChart(readings: List<Float>) {
    Canvas(Modifier.fillMaxWidth().height(140.dp)) {
        if (readings.size < 2) return@Canvas
        val min = (readings.min() - 10f).coerceAtLeast(40f)
        val max = (readings.max() + 10f).coerceAtMost(410f)
        val span = (max - min).coerceAtLeast(1f)
        val stepX = size.width / 11f
        readings.zipWithNext().forEachIndexed { i, (a, b) ->
            drawLine(
                color = Color(0xFF1565C0),
                start = Offset(i * stepX, size.height * (1 - (a - min) / span)),
                end = Offset((i + 1) * stepX, size.height * (1 - (b - min) / span)),
                strokeWidth = 4f,
            )
        }
    }
}
```

`android/app/src/main/kotlin/com/glucoedge/app/MainActivity.kt` (replace):
```kotlin
package com.glucoedge.app

import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.glucoedge.app.inference.TrendClassifier
import com.glucoedge.app.replay.CsvTraceLoader
import com.glucoedge.app.ui.GlucoEdgeTheme
import com.glucoedge.app.ui.MainScreen
import com.glucoedge.app.ui.MainViewModel
import com.glucoedge.app.ui.TraceSource
import java.io.InputStreamReader

class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels {
        object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                val (stream, label) = runCatching {
                    assets.open("real_trace.csv") to "real"
                }.getOrElse { assets.open("synthetic_trace.csv") to "synthetic" }
                val loaded = stream.use { CsvTraceLoader.load(InputStreamReader(it)) }
                return MainViewModel(
                    traceSource = TraceSource(loaded.readings, loaded.skippedRows, label),
                    classifierFactory = { model -> TrendClassifier.create(applicationContext, model) },
                ) as T
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val deviceLabel = if (Build.FINGERPRINT.contains("generic") ||
            Build.MODEL.contains("sdk_gphone") || Build.HARDWARE == "ranchu"
        ) "emulator" else "device"
        setContent { GlucoEdgeTheme { MainScreen(viewModel, deviceLabel) } }
    }
}
```

- [ ] **Step 6: Full unit suite + manual emulator verification**

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest :app:assembleDebug
~/Library/Android/sdk/platform-tools/adb install -r app/build/outputs/apk/debug/app-debug.apk
~/Library/Android/sdk/platform-tools/adb shell am start -n com.glucoedge.app/.MainActivity
sleep 8
~/Library/Android/sdk/platform-tools/adb exec-out screencap -p > /tmp/glucoedge_screen.png
```
Then tap Play via `adb shell input tap` (locate the Play button in the screenshot) or verify manually, wait ≥ 60 replay-seconds at 16×, take a second screenshot, and confirm: chart drawn, mg/dL value updating, a trend arrow with the exact caption "predicted trend, next 15 min", stats line showing latency + "emulator", disclaimer footer visible. Attach both screenshots to the task report.

- [ ] **Step 7: Commit**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
git add android/app/src/main/kotlin/com/glucoedge/app
git add android/app/src/test/kotlin/com/glucoedge/app/ui
git commit -m "Add main screen and view model wiring replay through on-device inference"
```

---

### Task 10: CI job + README section + final verification

**Files:**
- Modify: `.github/workflows/tests.yml`, `README.md`

**Interfaces:**
- Consumes: everything; the full local test story from Tasks 1–9.
- Produces: green CI on both jobs; README documents the app and the local-only parity gate.

- [ ] **Step 1: Add the android job to CI**

Append to `.github/workflows/tests.yml` under `jobs:` (keep the existing pytest job unchanged; match its indentation):
```yaml
  android:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: '17'
      - uses: android-actions/setup-android@v3
      - name: Build and unit-test the app
        working-directory: android
        run: ./gradlew --no-daemon :app:assembleDebug :app:testDebugUnitTest :app:checkNoInternetPermission
```

- [ ] **Step 2: Add the app section to README.md**

Insert after the existing conversion/results section (adapt placement to the README's current structure):
```markdown
## Android app (`android/`)

Kotlin + Jetpack Compose app that replays a CGM trace through the LiteRT
**CompiledModel** API entirely on-device. The manifest requests **no
INTERNET permission** — the OS itself guarantees no network calls for
inference, and CI fails if any dependency tries to merge that permission in.

- Bundled models: `trend_float.tflite` (default) and `trend_int8.tflite`
  (toggle in-app) — provenance and hashes in
  `android/app/src/main/assets/MODELS.md`.
- Replay data: a committed synthetic trace; optionally extract a real
  GlucoBench segment locally (never committed) with
  `python -m conversion.export_replay_trace`.
- Build: `cd android && ./gradlew :app:assembleDebug`
- Unit tests (run in CI): `./gradlew :app:testDebugUnitTest`
- **Parity tests (local-only gate, needs an emulator/device):**
  `./gradlew :app:connectedDebugAndroidTest` — proves the Kotlin inference
  path reproduces the Python benchmark's outputs (golden vectors) for both
  models, including round-and-clip INT8 input quantization.

The app shows measured on-device inference latency; numbers from an
emulator are labeled as such and are not device measurements.
```

- [ ] **Step 3: Full local verification sweep**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
venv/bin/python -m pytest -q                       # Python suite still green
cd android
JAVA_HOME=/opt/homebrew/opt/openjdk@17 ./gradlew :app:testDebugUnitTest :app:checkNoInternetPermission :app:assembleDebug
```
Expected: everything passes.

- [ ] **Step 4: Commit**

```bash
cd /Users/mohamednabil/Documents/vibe/GlucoEdge
git add .github/workflows/tests.yml README.md
git commit -m "Add Android CI job and README app section"
```

---

## Self-Review Notes

- **Spec coverage:** models bundled + toggle (T2, T8, T9); real+synthetic replay (T3); committed assets w/ provenance (T2); CompiledModel hot path w/ Interpreter one-shot metadata (T8); round+clip quantization incl. >240 test (T7); golden parity w/ sha256 pinning (T4, T8); no-INTERNET structural guarantee + CI enforcement (T1, T10); one screen + stats + disclaimer + rebuffering + trace label + emulator/device label (T9); error handling (T5 malformed rows, T9 empty trace/inference failure); reset-on-gap (T5); CI (T10). Latency measurement (T7, T9).
- **Known judgment calls:** the LiteRT TensorBuffer int8 API is behind a documented decision ladder in T8 rather than asserted, because the docs only show float examples; class-level parity is the hard gate, logit-level added when rung 1 holds.
- Emulator must be running for T8 Step 4 and T9 Step 6; T8 leaves it up for T9.
