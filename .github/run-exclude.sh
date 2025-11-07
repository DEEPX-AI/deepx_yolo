#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
PROJECT_ROOT=$(realpath -s "${SCRIPT_DIR}/../")

# release-excluded File Path
EXCLUDE_FILE=".github/release-excluded"

# Options
REMOVE_VENV=0

# Function to display help message
show_help() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --remove-venv       Remove venv folder if it exists in PROJECT_ROOT"
    echo ""
    echo "Description:"
    echo "  This script deletes files and directories listed in .github/release-excluded"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --remove-venv"
    echo "  $0 --help"
    echo ""
    exit 0
}

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            show_help
            ;;
        --remove-venv)
            REMOVE_VENV=1
            ;;
        *)
            echo "Error: Invalid option '$arg'"
            echo "Use --help or -h to see available options"
            exit 1
            ;;
    esac
done

pushd "$PROJECT_ROOT" || exit 1

# Remove venv if option is set
if [ "$REMOVE_VENV" -eq 1 ]; then
    if [ -d "${PROJECT_ROOT}/venv" ]; then
        echo "Removing venv folder..."
        deactivate || true
        rm -rf "${PROJECT_ROOT}/venv" && {
            echo "[Deleted] venv/"
        } || {
            echo "[Failed to delete] venv/"
            exit 1
        }
    else
        echo "[SKIP] venv folder not found"
    fi
fi

# File existence check
if [ ! -f "$EXCLUDE_FILE" ]; then
  echo "$EXCLUDE_FILE not found!"
  exit 1
fi

# Read and delete each path
while IFS= read -r path; do
  # Handle Windows CRLF line endings: remove \r
  path=$(echo "$path" | tr -d '\r')

  # Skip empty lines
  [ -z "$path" ] && continue

  # Wildcard pattern handling
  if [[ "$path" == *"*"* ]]; then
    # Expand glob patterns and delete matching files
    shopt -s nullglob
    matches=("${PROJECT_ROOT}/${path}")
    shopt -u nullglob
    
    if [ ${#matches[@]} -gt 0 ]; then
      for match in "${matches[@]}"; do
        echo "Deleting: ${match#${PROJECT_ROOT}/} ..."
        rm -rf "$match" && {
          echo "[Deleted] ${match#${PROJECT_ROOT}/}"
        } || {
          echo "[Failed to delete] ${match#${PROJECT_ROOT}/}"
          exit 1
        }
      done
    else
      echo "[SKIP] No matches found: $path"
    fi
  elif [ -e "${PROJECT_ROOT}/$path" ]; then
    echo "Deleting: $path ..."
    rm -rf "${PROJECT_ROOT}/$path" && {
        echo "[Deleted] $path"
    } || {
        echo "[Failed to delete] $path"
        exit 1
    }
  else
    echo "[SKIP] Not found: $path"
  fi
done < "$EXCLUDE_FILE"

popd || exit 1
exit 0
