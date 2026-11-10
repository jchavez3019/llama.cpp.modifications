# Building llama.cpp for the laptop and the Pixel Fold

This page explains how this repository is built, and how to make the two builds used by the radix tree KV cache work: a CUDA build for the laptop, and a CPU build for the Pixel Fold. It also covers the tools around the builds: CLion, `adb`, and the Conda environment for the Python benchmark.

For Phase 0, the result of following this page is:

| Device | Build directory | Binaries |
| --- | --- | --- |
| Laptop (RTX 3060 Laptop, 6 GB) | `build-laptop-cuda-release/bin/` | `llama-server`, `llama-bench`, plus the shared libraries they load |
| Pixel Fold (Tensor G2) | `build-android-arm64-release/bin/` | `llama-server`, `llama-bench` (static, no libraries needed) |

## The big picture

llama.cpp is built in layers. Each layer is a CMake target, and each one depends only on the layers below it.

```
+---------------------------------------------------------------+
| tools: llama-server, llama-bench, llama-cli, ...  (tools/)    |
+---------------------------------------------------------------+
| llama-common: argument parsing, sampling, chat (common/)      |
+---------------------------------------------------------------+
| llama: models, KV cache, batching, the C API (src/, include/) |
+---------------------------------------------------------------+
| ggml: tensors and compute graphs (ggml/)                       |
|   ggml-base + one library per backend:                        |
|   ggml-cpu, ggml-cuda, ggml-vulkan, ggml-metal, ...           |
+---------------------------------------------------------------+
```

*The KV cache work happens in `src/` (`llama-kv-cache.cpp`) and `tools/server/`. The ggml layer decides which hardware runs the math.*

The important point for this project: **the hardware choice is made when you build, not when you run**. A build without `GGML_CUDA=ON` cannot use the GPU, whatever flags you pass to `llama-server` later.

## How CMake works here

CMake does not compile anything itself. It is a build-system generator: it reads `CMakeLists.txt` and writes instructions for a lower-level build tool. Here that tool is **Ninja**, a small program that reads `build.ninja` and runs the compiler commands in it, in parallel, and only for files that changed. Make does the same job, but Ninja starts faster and uses all cores by default.

A build has up to three steps. Each step is one command, and each has a preset form that does the same thing with fewer flags.

| Step | Raw command | Preset command | What it does | When to run it |
| --- | --- | --- | --- | --- |
| 1. Configure | `cmake -B <dir> -G Ninja -D...` | `cmake --preset <name>` | Checks the compilers and options, stores them in `<dir>/CMakeCache.txt`, and writes `<dir>/build.ninja`. Compiles nothing. | Once per build directory, and again after you change an option. CLion runs it when it loads or reloads the project. |
| 2. Build | `cmake --build <dir> --target ...` | `cmake --build --preset <name>` | Runs Ninja in `<dir>`. Ninja compiles what changed and writes the programs to `<dir>/bin/`. | After every code change. CLion's **Build** button (Ctrl+F9) runs it. |
| 3. Install | `cmake --install <dir> --prefix <path>` | (none) | Copies the finished programs, shared libraries, and headers from `<dir>` into a clean `bin/`, `lib/`, `include/` layout under `<path>`. Compiles nothing. | Only to collect files for another machine. Our presets never need it (see Step 2). |

Most build problems come from mixing up steps 1 and 2.

Options given with `-D` in step 1 are stored in `CMakeCache.txt` and **stay there**. If you configure the same directory again with different `-D` values, only those values change. If you leave an option out, the old value is kept, not reset to the default. To start clean, delete the build directory.

### Building again

Each build directory is one independent build. Building again in the same directory **replaces** the old programs in `<dir>/bin/`: Ninja recompiles only the files that changed (and the files that include them) and then relinks. There is no history of old binaries. Copy a binary elsewhere first if you want to keep it.

Different directories never touch each other. `build/` (from the raw command), `build-laptop-cuda-release/` (from the preset), and `cmake-build-debug/` (from CLion's Debug profile) are three separate builds of the same source, with their own options and binaries. Each one takes 0.5-1.5 GB, so delete the ones you do not use.

### The option tree

`CMakeLists.txt` at the root defines the `LLAMA_*` options and then adds the subdirectories. `ggml/CMakeLists.txt` defines the `GGML_*` options. The options this project uses:

| Option | Default | Meaning |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | none (no optimization flags) | `Release` = `-O3 -DNDEBUG`. `Debug` = `-O0 -g`, often 5-20x slower on the CPU. `RelWithDebInfo` = `-O2 -g`. |
| `BUILD_SHARED_LIBS` | `ON` on Linux and Android | `ON` builds `libllama.so`, `libggml*.so` and links the tools to them. `OFF` puts everything into each executable. |
| `LLAMA_BUILD_TESTS` / `LLAMA_BUILD_EXAMPLES` / `LLAMA_BUILD_TOOLS` / `LLAMA_BUILD_SERVER` | `ON` | Which groups of targets exist. Tests and examples are turned off in our presets to save build time. |
| `LLAMA_OPENSSL` | `ON` | HTTPS support (for downloading models with `-hf`). Turned off for Android, where OpenSSL is not available in the NDK. |
| `GGML_CUDA` | `OFF` | Build the CUDA backend (`ggml/src/ggml-cuda/`). |
| `CMAKE_CUDA_ARCHITECTURES` | depends on the CUDA version | GPU generations to compile kernels for. `86` = RTX 30xx (Ampere). One value builds much faster than the default list. |
| `GGML_NATIVE` | `ON`, but `OFF` when cross-compiling | Compile the CPU code for the machine that runs the compiler (`-march=native`). Wrong for a phone build, because the compiler runs on the laptop. |
| `GGML_CPU_ARM_ARCH` | empty | ARM CPU target for the CPU backend, for example `armv8.2-a+dotprod+fp16` ([ggml/CMakeLists.txt](../../ggml/CMakeLists.txt) line 184). |
| `GGML_OPENMP` | `ON` | Use OpenMP threads. Off for Android: the NDK OpenMP library is a shared library that would have to be pushed too. ggml then uses its own thread pool. |
| `GGML_LLAMAFILE` | `ON` for most targets | The llamafile matrix multiply kernels. Off for Android, as in [docs/android.md](../android.md). |

Old names such as `LLAMA_CUDA` or `LLAMA_NATIVE` still work but print a warning. They are mapped to the `GGML_*` names (see `llama_option_depr` in the root `CMakeLists.txt`).

### Targets

A target is one thing CMake can build. `cmake --build <dir>` with no `--target` builds every target that the options enable. With `--target llama-server llama-bench`, Ninja builds only these two and what they depend on:

- `llama-server` is `tools/server/main.cpp`, linked to `llama-server-impl` (the HTTP layer), which is linked to `server-context` (slots, the prompt cache, tasks).
- `llama-bench` is `tools/llama-bench/main.cpp`, linked to `llama-bench-impl`.
- Both link `llama-common`, `llama`, and the `ggml` libraries.

The executables go to `<dir>/bin/`.

### Presets

A preset is a named set of configure options stored in a JSON file, so you do not have to retype long `-D` lists.

- `CMakePresets.json` is part of the repository. It holds hidden building blocks (`base`, `release`, `debug`, `static`, `vulkan`, ...) and some public presets for other platforms.
- `CMakeUserPresets.json` is for local presets. CMake reads it automatically next to `CMakePresets.json`. This repository has one at the root with our two presets. It is listed in `.git/info/exclude`, so git ignores it.

Our presets inherit two blocks from `CMakePresets.json`:

- `base`: Ninja generator; build directory `build-<preset name>`; `compile_commands.json` for editors.
- `release`: `CMAKE_BUILD_TYPE=Release`.

The file has two kinds of presets, one per step:

- A **configure preset** (`configurePresets`) stands for step 1. It holds the build directory, the generator, and every `-D` option. `cmake --preset laptop-cuda-release` is the same as typing `cmake -B build-laptop-cuda-release -G Ninja` plus all its `-D` options.
- A **build preset** (`buildPresets`) stands for step 2. It names its configure preset (so CMake knows the directory) and the target list. `cmake --build --preset laptop-cuda-release` is the same as `cmake --build build-laptop-cuda-release --target llama-server llama-bench`.

```bash
cmake --list-presets                              # list configure presets
cmake --preset laptop-cuda-release                # step 1: configure -> build-laptop-cuda-release/
cmake --build --preset laptop-cuda-release -j 12  # step 2: build llama-server and llama-bench
```

Options after the preset name, such as `-j 12`, are passed on to that step.

## One-time setup

Do these once per machine, before the first build.

### Put adb on your PATH

`adb` (Android Debug Bridge) is the tool that talks to the phone over USB. Android Studio installed it at `~/Android/Sdk/platform-tools/adb`, but that directory is not on your `PATH`, so the shell cannot find `adb` by name.

Add these lines to the end of `~/.bashrc`:

```bash
export ANDROID_HOME="$HOME/Android/Sdk"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
```

Then load them in the open terminal, and check:

```bash
source ~/.bashrc
which adb            # /home/jorgejc2/Android/Sdk/platform-tools/adb
adb devices          # the phone shows as "device"; "unauthorized" means: accept the prompt on the phone
```

Notes:

- `~/.bashrc` is read by terminals only. Programs started from the desktop menu (for example CLion) read `~/.profile`. To make CLion's tools see `adb` too, put the same two lines in `~/.profile` and log out and in again.
- Prefer this `adb` to `sudo apt install adb`. The apt version is older, and two different `adb` versions on one machine keep restarting each other's server ("adb server version doesn't match this client").
- Until you do this, use the full path `~/Android/Sdk/platform-tools/adb`. The benchmark scripts accept `--adb <path>`.

### Fix the system ninja

This laptop has two `ninja` programs, and the one found first is broken:

| Path | Installed by | State |
| --- | --- | --- |
| `/usr/local/bin/ninja` | `sudo pip install ninja` (the PyPI package `ninja` 1.11.1.1, November 2023) | Broken. It is a small Python script that starts the real binary inside the pip package, but that binary is missing its execute permission. |
| `/usr/bin/ninja` | `apt` (the package `ninja-build` 1.10.1) | Works. |

`/usr/local/bin` comes before `/usr/bin` on the `PATH`, so the shell and CMake pick the broken one. A configure then fails with `PermissionError: [Errno 13]`, and then with the misleading `CMAKE_C_COMPILER not set`.

The apt `ninja` is already installed, so `sudo apt install ninja-build` is not needed. Remove the pip wrapper, and the working `/usr/bin/ninja` is found:

```bash
sudo python3 -m pip uninstall ninja   # removes /usr/local/bin/ninja and the pip package
hash -r                               # make the open shell forget the old path
which ninja && ninja --version        # /usr/bin/ninja, 1.10.1
```

Ninja 1.10 is new enough for this project. Until you do this, the Conda environment below is the workaround: it has its own `ninja` that comes first on the `PATH` while the environment is active.

A build directory stores the full path of the `ninja` it was configured with (`CMAKE_MAKE_PROGRAM` in `CMakeCache.txt`), and every later build uses that path. `build-laptop-cuda-release/` and `build-android-arm64-release/` were configured in the Conda environment, so they use `~/anaconda3/envs/llama-orch-bench/bin/ninja`, also from CLion. That works as long as the environment exists. If you delete it, delete these build directories as well and configure again.

### Create the Conda environment

The benchmark is Python. The Conda environment `llama-orch-bench` keeps its packages apart from the system Python. It is defined in [tools/server/bench/orchestration-bench/environment.yml](../../tools/server/bench/orchestration-bench/environment.yml): Python 3.11, `requests`, `pytest`, and `ninja`, plus `nvidia-ml-py` (GPU telemetry) and `perfetto` (phone trace analysis) from pip.

```bash
source ~/anaconda3/etc/profile.d/conda.sh     # makes the "conda" command available in this shell
conda env create -f tools/server/bench/orchestration-bench/environment.yml   # once
conda activate llama-orch-bench               # in every new terminal
```

To update the environment after `environment.yml` changes: `conda env update -f tools/server/bench/orchestration-bench/environment.yml --prune`.

### Check the disk space

Check the free space with `df -h /`. A CUDA build directory takes about 0.6-1.5 GB, an Android one about 1.2 GB, CLion's Debug build about 0.9 GB, and the three models about 2.3 GB. Delete old build directories that you do not use (for example `rm -rf build-laptop-cuda-release` before a clean rebuild).

## Step 1: build for the laptop

The Phase 0 plan gives this command:

```bash
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-bench
```

What each part means:

| Part | Meaning |
| --- | --- |
| `cmake -B build` | Configure, and put all generated files in the directory `build/`. |
| `-G Ninja` | Use Ninja as the build tool (faster than Make, and it uses all cores by default). |
| `-DGGML_CUDA=ON` | Compile the CUDA backend, so model layers can run on the GPU. |
| `-DCMAKE_CUDA_ARCHITECTURES=86` | Compile the CUDA kernels only for compute capability 8.6 (the RTX 3060). Without it, CMake compiles for several GPU generations, which takes much longer. |
| `-DCMAKE_BUILD_TYPE=Release` | Full optimization, no debug asserts from the standard library. |
| `cmake --build build` | Run the build in `build/`. |
| `--target llama-server llama-bench` | Build only these two programs and their libraries. |

The first line is step 1 (configure) and the second is step 2 (build). The preset pair below does the same two steps, with three differences:

- the build directory is `build-laptop-cuda-release/` instead of `build/`, so both can exist side by side;
- `CMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc`, because `nvcc` is not on the `PATH`;
- tests and examples off.

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate llama-orch-bench
cmake --preset laptop-cuda-release
cmake --build --preset laptop-cuda-release -j 12
```

`-j 12` limits Ninja to 12 parallel jobs. The CUDA template files use about 1-2 GB of RAM each, so all 20 threads at once can run out of memory. The build has 336 steps and took about 7 minutes on this laptop. Most of the time goes to the CUDA kernel files.

GPU use is then a run-time flag. `-ngl` sets how many model layers go to VRAM. The default is `auto` (as many as fit), and `-ngl 99` or `-ngl all` puts all layers on the GPU. The benchmark passes `-ngl 99`, so a run never silently falls back to a partial offload.

## Step 2: build for the phone

The phone build is a **cross-compile**: the compiler runs on the laptop (x86-64) and produces code for the phone (ARM64, Android). The Android NDK provides that compiler (clang) and the Android system libraries.

The Phase 0 plan gives:

```bash
export ANDROID_NDK=~/Android/Sdk/ndk/26.3.11579264
cmake -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \
  -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-android --target llama-server llama-bench
cmake --install build-android --prefix build-android/install
```

What each part means:

| Part | Meaning |
| --- | --- |
| `export ANDROID_NDK=...` | Shell variable with the NDK location, used in the next line. NDK 26.3 is the version installed by Android Studio. |
| `-DCMAKE_TOOLCHAIN_FILE=.../android.toolchain.cmake` | A CMake script from the NDK. It replaces the laptop's GCC with the NDK clang, and sets the Android system root. This is what makes the build a cross-compile. |
| `-DANDROID_ABI=arm64-v8a` | Produce 64-bit ARM code (all modern phones). |
| `-DANDROID_PLATFORM=android-28` | The oldest Android version (9) the binaries run on. Newer versions run them too. The phone runs Android 17. |
| `-DGGML_NATIVE=OFF` | Do not use `-march=native`: "native" would mean the laptop CPU. |
| `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16` | Compile for the Tensor G2 cores: ARMv8.2 with the dot-product instructions (`asimddp`, the main speed-up for quantized models) and half-precision math (`fphp`). The G2 has no `i8mm` and no SVE, so those are left out. A binary with them would crash with "illegal instruction". |
| `-DGGML_OPENMP=OFF` | Use ggml's own thread pool. The NDK OpenMP is a separate shared library that would also have to be pushed. |
| `-DGGML_LLAMAFILE=OFF` | Turn off the llamafile kernels, as in [docs/android.md](../android.md). |
| `-DLLAMA_OPENSSL=OFF` | No HTTPS client. Models are pushed with `adb`, not downloaded on the phone. |
| `cmake --build build-android --target ...` | Step 2: compile the two programs with the NDK clang. |
| `cmake --install ... --prefix build-android/install` | Step 3: copy the binaries and the shared libraries (`.so`) to `install/bin` and `install/lib`, ready to push. |

This command list uses all three steps because it builds shared libraries (the default). `llama-server` then needs `libllama.so`, `libggml*.so`, and others next to it on the phone, and step 3 collects them in one place.

The preset pair below does steps 1 and 2 into `build-android-arm64-release/`, and adds `BUILD_SHARED_LIBS=OFF`. Then each executable contains all the libraries, so there is nothing to collect and no `LD_LIBRARY_PATH` to set on the phone. You push `bin/llama-server` directly, and step 3 is not needed.

```bash
cmake --preset android-arm64-release
cmake --build --preset android-arm64-release -j 6
```

The binaries are large (about 190 MB for `llama-server`). The NDK toolchain always adds `-g` (debug information). It does not change the speed. To push smaller files, strip a copy first: `~/Android/Sdk/ndk/26.3.11579264/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip -o /tmp/llama-server build-android-arm64-release/bin/llama-server`.

### Push and run on the phone

`/data/local/tmp` is the one directory where the `adb shell` user can write files and run programs without root.

```bash
adb shell mkdir -p /data/local/tmp/llama.cpp/models
adb push build-android-arm64-release/bin/llama-server build-android-arm64-release/bin/llama-bench /data/local/tmp/llama.cpp/
adb push ~/models/*.gguf /data/local/tmp/llama.cpp/models/
adb shell "cd /data/local/tmp/llama.cpp && ./llama-server --version"
```

Run the server on the phone and reach it from the laptop:

```bash
adb forward tcp:8080 tcp:8080     # laptop port 8080 -> phone port 8080
adb shell "cd /data/local/tmp/llama.cpp && taskset f0 ./llama-server -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf -t 4 -c 4096 --port 8080"
curl http://127.0.0.1:8080/health   # in a second terminal on the laptop
```

`taskset f0` limits the process to CPU cores 4-7 (mask `0xf0`): the two Cortex-X1 and two Cortex-A78 cores. `-t 4` uses one thread per core. The four small Cortex-A55 cores (0-3) are slower and make every thread wait for the slowest one. See [phase_0_baseline.md](phase_0_baseline.md) for the measured thread sweep.

## CLion

CLion runs the same CMake steps for you. Its settings decide which options it configures with.

### What your existing CLion build is

CLion's default profile is called **Debug**, and it builds into `cmake-build-debug/`. Its `CMakeCache.txt` shows:

| Option | Value in `cmake-build-debug` | Effect |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Debug` | No optimization: slow, but good for stepping through code |
| `GGML_CUDA` | `OFF` | CPU only, the GPU is never used |
| `GGML_NATIVE` | `ON` | Optimized for this laptop's CPU (fine on the laptop only) |
| `BUILD_SHARED_LIBS` | `ON` | Tools load `libllama.so` and friends from `bin/` |
| Tests and examples | `ON` | Many extra targets |

So the `llama-server` and `llama-bench` you built in CLion work, but they are **not usable for benchmarks**: they are Debug and CPU-only. Keep that profile for debugging (it is the right one for setting breakpoints in `server-context.cpp`), and use the presets for measurements.

To check any build directory yourself:

```bash
grep -E "^(CMAKE_BUILD_TYPE|GGML_CUDA|GGML_NATIVE|GGML_CPU_ARM_ARCH|BUILD_SHARED_LIBS):" <dir>/CMakeCache.txt
<dir>/bin/llama-server --version          # build number, commit, and compiler
<dir>/bin/llama-bench -m MODEL -n 0 -p 64   # the "backend" column shows CUDA or CPU
```

### Use the presets in CLion

CLion loads `CMakePresets.json` and `CMakeUserPresets.json` by itself and shows each configure preset as a CMake profile.

1. Open **Settings > Build, Execution, Deployment > CMake**.
2. In the profile list on the left, find **laptop-cuda-release** and **android-arm64-release**. Preset profiles are greyed out (read-only) until you enable them. Select one and tick **Enable profile**. Leave the hidden building blocks (`base`, `release`, ...) disabled.
3. Keep the **Debug** profile enabled if you want to keep debugging with it.
4. Click **OK**. CLion configures each enabled profile into the preset's directory (`build-laptop-cuda-release/`, `build-android-arm64-release/`).
5. In the toolbar, the configuration switcher now has a profile part. Choose the `llama-server` or `llama-bench` run configuration, and the **laptop-cuda-release** profile, then **Build** (Ctrl+F9).

### What CLion runs when you enable a profile

When you enable or reload a profile, CLion runs **step 1 (configure) only**, with its own bundled CMake (`/snap/clion/<version>/bin/cmake/...`, CMake 4.4). Nothing is compiled until you press **Build**, which runs step 2 for the selected target.

| Profile | Command CLion runs | Same as |
| --- | --- | --- |
| Debug | `cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MAKE_PROGRAM=<CLion's ninja> -G Ninja -S . -B cmake-build-debug` | A raw configure with no options except the build type |
| laptop-cuda-release | `cmake -DCMAKE_BUILD_TYPE=Release --preset laptop-cuda-release -S . -B build-laptop-cuda-release` | `cmake --preset laptop-cuda-release` |
| android-arm64-release | `cmake -DCMAKE_BUILD_TYPE=Release --preset android-arm64-release -S . -B build-android-arm64-release` | `cmake --preset android-arm64-release` |

`-S` is the source directory and `-B` the build directory. Both come from the preset anyway, as does `-DCMAKE_BUILD_TYPE=Release` (CLion passes the profile's build type again). So the two preset profiles run the same configure as the commands in Step 1 and Step 2 above, only into the preset directories.

Messages in the output:

- **"Detecting C compiler ABI info" and the rest of the compiler checks:** CMake caches these per CMake version (`<dir>/CMakeFiles/<version>/`). The preset directories were first configured in the terminal with CMake 3.26 (`~/.local/bin/cmake`), so CLion's CMake 4.4 checks the compilers again once. The options and the compiled files stay. Later builds can be started from either tool. The configure step is just repeated whenever the other CMake version touches the directory.
- **`ccache not found`:** ccache is an optional compiler cache that makes rebuilds after a clean faster. It is safe to ignore. `sudo apt install ccache` turns it on.
- **`Could NOT find NCCL`:** NCCL is for multi-GPU setups. With one GPU it is not needed.
- **`Compatibility with CMake < 3.10 will be removed` (Android only):** NDK 26's `android.toolchain.cmake` declares an old minimum CMake version, and CMake 4 warns about that every time it reads the file. It is a warning for the NDK authors and has no effect on the build. A newer NDK (27 or later) does not print it.

Notes:

- You cannot edit a preset's options in the CLion dialog. Change `CMakeUserPresets.json` instead, then use **Tools > CMake > Reload CMake Project**.
- The Android profile builds fine in CLion, but CLion cannot run or debug the result on the laptop. Push and run it with `adb` as above.
- A build directory remembers the `cmake` and `ninja` that created it. If you configure `build-laptop-cuda-release/` in the terminal and then in CLion (or the other way round) and the build fails with a generator or `CMAKE_MAKE_PROGRAM` error, delete the directory and let one tool own it.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `CMAKE_C_COMPILER not set` after a `PermissionError` | Broken pip `ninja` in `/usr/local/bin` | Run `sudo python3 -m pip uninstall ninja`, or activate `llama-orch-bench`. Delete the half-configured build directory first. |
| Build fails with `ninja: not found` or a `CMAKE_MAKE_PROGRAM` error | The `ninja` path stored in `CMakeCache.txt` no longer exists (for example, the Conda environment was removed) | Delete the build directory and configure again. |
| `No CMAKE_CUDA_COMPILER could be found` | `nvcc` is not on the `PATH` | Use the preset (it sets the full path), or add `/usr/local/cuda-12.6/bin` to `PATH`. |
| CUDA build is killed, or the laptop freezes | Too many parallel `nvcc` jobs | Use `-j 8` or `-j 12`. |
| `illegal instruction` on the phone | Built for a CPU feature the G2 does not have (`i8mm`, SVE) | Check `GGML_CPU_ARM_ARCH` in `build-android-arm64-release/CMakeCache.txt`. |
| `CANNOT LINK EXECUTABLE ... libllama.so not found` on the phone | Shared build pushed without its libraries | Use the static preset, or push `lib/` and run with `LD_LIBRARY_PATH=lib`. |
| `adb: command not found` | `platform-tools` is not on the `PATH` | See "Put adb on your PATH". |
| `adb devices` shows `unauthorized` | USB debugging not accepted yet | Unlock the phone and accept the prompt. |
| Server on the phone dies when memory is low | Android's low-memory killer | Close apps. Check `adb shell grep MemAvailable /proc/meminfo` is at least 2 GB. |
