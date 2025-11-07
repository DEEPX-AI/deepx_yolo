#!/bin/bash
SCRIPT_DIR=$(realpath "$(dirname "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR")
PROJECT_NAME=$(basename "$SCRIPT_DIR")
VENV_PATH="$PROJECT_ROOT/venv*"

pushd "$PROJECT_ROOT" >&2

# color env settings
source ${PROJECT_ROOT}/scripts/color_env.sh
source ${PROJECT_ROOT}/scripts/common_util.sh

ENABLE_DEBUG_LOGS=0

show_help() {
    echo -e "Usage: ${COLOR_CYAN}$(basename "$0") [OPTIONS]${COLOR_RESET}"
    echo -e ""
    echo -e "Options:"
    echo -e "  ${COLOR_GREEN}[-v|--verbose]${COLOR_RESET}                        Enable verbose (debug) logging"
    echo -e "  ${COLOR_GREEN}[-h|--help]${COLOR_RESET}                           Display this help message and exit"
    echo -e ""
    
    if [ "$1" == "error" ] && [[ ! -n "$2" ]]; then
        print_colored_v2 "ERROR" "Invalid or missing arguments."
        exit 1
    elif [ "$1" == "error" ] && [[ -n "$2" ]]; then
        print_colored_v2 "ERROR" "$2"
        exit 1
    elif [[ "$1" == "warn" ]] && [[ -n "$2" ]]; then
        print_colored_v2 "WARNING" "$2"
        return 0
    fi
    exit 0
}

# cleanup_common_files() {
#     print_colored_v2 "INFO" "Cleanup common files..."
#     delete_symlinks "$PROJECT_ROOT"
#     delete_symlinks "${VENV_PATH}"
#     delete_dir "${VENV_PATH}"
# }

cleanup_project_specific_files() {
    print_colored_v2 "INFO" "Cleanup ${PROJECT_NAME} specific files..."

    EXCLUDE_FILE="${PROJECT_ROOT}/.github/output-results"
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
}

main() {
    print_colored_v2 "INFO" "Cleanup ${PROJECT_NAME} ..."

    # Remove symlinks for 'Common' Rules
    # cleanup_common_files

    # Cleanup the project specific files
    cleanup_project_specific_files

    print_colored_v2 "SUCCESS" "[OK] Cleanup ${PROJECT_NAME} done"
}

# parse args
for i in "$@"; do
    case "$1" in
        -v|--verbose)
            ENABLE_DEBUG_LOGS=1
            ;;
        -h|--help)
            show_help
            ;;
        *)
            show_help "error" "Invalid option '$1'"
            ;;
    esac
    shift
done

main

popd >&2

exit 0
