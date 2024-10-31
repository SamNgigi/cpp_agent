#!/usr/bin/env bash

# Get the directory name
dir_name=$(basename "$(pwd)")

# Define paths for executables (Git Bash handles the .exe extension automatically)
exe_path="build/Debug/$dir_name"
test_exe_path="build/Debug/${dir_name}_test"

# Function to clean build artifacts directory
clean() {
    echo "Cleaning build directory..."
    if [ -d "build" ]; then
        echo "Removing existing build directory..."
        rm -rf build || { echo "Failed to remove build directory."; exit 1; }
    fi
    echo "Creating new build directory."
    mkdir build || { echo "Failed to create build directory."; exit 1; }
    echo "Clean completed successfully."
}

# Function to build project
build() {
    mkdir -p build || { echo "Failed to create build directory."; exit 1; }
    pushd build > /dev/null || { echo "Failed to enter build directory."; exit 1; }

    # Try to find vcpkg in common Git Bash locations
    vcpkg_paths=(
        "$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"
        "/c/vcpkg/scripts/buildsystems/vcpkg.cmake"
        "/c/Users/$USER/vcpkg/scripts/buildsystems/vcpkg.cmake"
    )

    # Find the first existing vcpkg path
    vcpkg_path=""
    for path in "${vcpkg_paths[@]}"; do
        if [ -f "$path" ]; then
            vcpkg_path="$path"
            break
        fi
    done

    # Configure CMake with or without vcpkg
    if [ -n "$vcpkg_path" ]; then
        echo "Using vcpkg toolchain at: $vcpkg_path"
        cmake -DCMAKE_TOOLCHAIN_FILE="$vcpkg_path" .. || {
            echo "CMake configuration failed."
            popd > /dev/null
            exit 1
        }
    else
        echo "No vcpkg toolchain found, using default CMake configuration"
        cmake .. || {
            echo "CMake configuration failed."
            popd > /dev/null
            exit 1
        }
    fi

    # Build the project
    cmake --build . || {
        echo "Build failed."
        popd > /dev/null
        exit 1
    }

    popd > /dev/null
    echo "Build completed successfully."
}

# Function to run the executable
run() {
    if [ ! -f "${exe_path}.exe" ] && [ ! -f "$exe_path" ]; then
        echo "Executable not found. Building the project..."
        build || exit 1
    fi
    echo "Running executable..."
    "$exe_path" "$@"
}

# Function to run tests
test() {
    if [ ! -f "${test_exe_path}.exe" ] && [ ! -f "$test_exe_path" ]; then
        echo "Test executable not found. Building the project..."
        build || exit 1
    fi
    echo "Running tests..."
    "$test_exe_path"
}

# Main logic
actions=()
app_args=()
parsing_actions=true

# Parse arguments
for arg in "$@"; do
    if $parsing_actions; then
        case $arg in
            clean|build|run|test)
                actions+=("$arg")
                ;;
            *)
            parsing_actions=false
            app_args+=("$arg")
            ;;
        esac
    else
        app_args+=("$arg")
    fi
done

# If no action default to "run"
if [ ${#actions[@]} -eq 0 ]; then
    actions+=("run")
fi

# Execute actions
for action in "${actions[@]}"; do
    case $action in
        build)
            build
            ;;
        clean)
            clean
            ;;
        run)
            run "${app_args[@]}"
            ;;
        test)
            test
            ;;
        *)
            echo "Unknown action: $action"
            echo "Valid actions: build, clean, run, test"
            exit 1
            ;;
    esac
done
