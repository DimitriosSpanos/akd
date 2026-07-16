#!/usr/bin/env bash
# Download the six CIFAR-100 teacher checkpoints expected by AKD.
# Uses the maintained mdistiller GitHub release mirror because the original
# shape2prog.csail.mit.edu server is no longer accepting connections.

set -Eeuo pipefail

readonly RELEASE_URL="https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar"
readonly ARCHIVE_NAME="cifar_teachers.tar"

FORCE=0
VERIFY=1

usage() {
    cat <<'USAGE'
Usage: fetch_pretrained_teachers_fixed.sh [--force] [--no-verify]

Options:
  --force       Replace checkpoints that already exist.
  --no-verify   Skip PyTorch architecture/state-dict compatibility checks.
  -h, --help    Show this help message.

The script may be run from the AKD repository root, from AKD/scripts, or from
another directory if the script itself is stored inside the AKD repository.
USAGE
}

while (($#)); do
    case "$1" in
        --force) FORCE=1 ;;
        --no-verify) VERIFY=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

log()  { printf '\033[1;34m[AKD]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }

command -v tar >/dev/null 2>&1 || die "tar is required but was not found."

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

is_repo_root() {
    [[ -f "$1/train_cifar_student.py" && -f "$1/train_cifar_teacher.py" && -d "$1/models/cifar" ]]
}

find_repo_root() {
    local candidate dir
    for candidate in "$PWD" "$SCRIPT_DIR" "$SCRIPT_DIR/.."; do
        candidate="$(cd -- "$candidate" 2>/dev/null && pwd -P)" || continue
        if is_repo_root "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    for dir in "$PWD" "$SCRIPT_DIR"; do
        dir="$(cd -- "$dir" 2>/dev/null && pwd -P)" || continue
        while [[ "$dir" != "/" ]]; do
            if is_repo_root "$dir"; then
                printf '%s\n' "$dir"
                return 0
            fi
            dir="$(dirname -- "$dir")"
        done
    done
    return 1
}

REPO_ROOT="$(find_repo_root)" || die \
    "Could not locate the AKD repository. Put this script in AKD/scripts or run it from the AKD repository."

readonly REPO_ROOT
readonly MODEL_ROOT="$REPO_ROOT/save/models"
readonly CACHE_DIR="$REPO_ROOT/.cache/akd"
readonly ARCHIVE_PATH="$CACHE_DIR/$ARCHIVE_NAME"

# directory:model-constructor pairs
readonly TEACHERS=(
    "wrn_40_2_vanilla:wrn_40_2"
    "resnet56_vanilla:resnet56"
    "resnet110_vanilla:resnet110"
    "resnet32x4_vanilla:resnet32x4"
    "vgg13_vanilla:vgg13"
    "ResNet50_vanilla:ResNet50"
)

mkdir -p "$MODEL_ROOT" "$CACHE_DIR"
log "AKD repository: $REPO_ROOT"
log "Checkpoint destination: $MODEL_ROOT"

all_present=1
for spec in "${TEACHERS[@]}"; do
    dir="${spec%%:*}"
    target="$MODEL_ROOT/$dir/ckpt_epoch_240.pth"
    if [[ ! -s "$target" ]]; then
        all_present=0
        break
    fi
done

if ((all_present && ! FORCE)); then
    ok "All six checkpoint files already exist; download is not needed."
else
    tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/akd-teachers.XXXXXXXX")"
    cleanup() { rm -rf -- "$tmp_dir"; }
    trap cleanup EXIT INT TERM

    if ((FORCE)) || [[ ! -s "$ARCHIVE_PATH" ]]; then
        partial="$ARCHIVE_PATH.part"
        rm -f -- "$partial"
        log "Downloading CIFAR-100 teacher archive from the mdistiller GitHub release..."

        if command -v curl >/dev/null 2>&1; then
            curl -fL --retry 5 --retry-delay 2 --connect-timeout 20 \
                --output "$partial" "$RELEASE_URL" \
                || die "Download failed. Check access to github.com and try again."
        elif command -v wget >/dev/null 2>&1; then
            wget --tries=5 --timeout=20 --output-document="$partial" "$RELEASE_URL" \
                || die "Download failed. Check access to github.com and try again."
        else
            die "Install either curl or wget, then run this script again."
        fi

        [[ -s "$partial" ]] || die "The downloaded archive is empty."
        mv -f -- "$partial" "$ARCHIVE_PATH"
        ok "Downloaded: $ARCHIVE_PATH"
    else
        log "Using cached archive: $ARCHIVE_PATH"
    fi

    log "Checking and extracting archive..."
    tar -tf "$ARCHIVE_PATH" >/dev/null 2>&1 \
        || die "The cached archive is invalid. Delete '$ARCHIVE_PATH' and rerun the script."
    tar -xf "$ARCHIVE_PATH" -C "$tmp_dir"

    copied=0
    skipped=0
    for spec in "${TEACHERS[@]}"; do
        dir="${spec%%:*}"
        target_dir="$MODEL_ROOT/$dir"
        target="$target_dir/ckpt_epoch_240.pth"

        if [[ -s "$target" && $FORCE -eq 0 ]]; then
            ok "Keeping existing $dir/ckpt_epoch_240.pth"
            ((skipped+=1))
            continue
        fi

        source_file="$(find "$tmp_dir" -type f \
            -path "*/$dir/ckpt_epoch_240.pth" -print -quit)"
        [[ -n "$source_file" && -s "$source_file" ]] \
            || die "Archive does not contain $dir/ckpt_epoch_240.pth"

        mkdir -p "$target_dir"
        temp_target="$target.tmp.$$"
        cp -- "$source_file" "$temp_target"
        mv -f -- "$temp_target" "$target"
        ok "Installed $dir/ckpt_epoch_240.pth"
        ((copied+=1))
    done

    log "Installed $copied checkpoint(s); kept $skipped existing checkpoint(s)."
fi

# Always ensure the expected files are present and non-empty.
for spec in "${TEACHERS[@]}"; do
    dir="${spec%%:*}"
    target="$MODEL_ROOT/$dir/ckpt_epoch_240.pth"
    [[ -s "$target" ]] || die "Missing or empty checkpoint: $target"
done

if ((VERIFY)); then
    if command -v python3 >/dev/null 2>&1 && python3 -c 'import torch' >/dev/null 2>&1; then
        log "Verifying all checkpoints against AKD model definitions..."
        (
            cd "$REPO_ROOT"
            python3 - "$REPO_ROOT" <<'PY'
import gc
import os
import sys

root = os.path.abspath(sys.argv[1])
sys.path.insert(0, root)

import torch
from models.cifar import model_dict_teacher

teachers = [
    ("wrn_40_2_vanilla", "wrn_40_2"),
    ("resnet56_vanilla", "resnet56"),
    ("resnet110_vanilla", "resnet110"),
    ("resnet32x4_vanilla", "resnet32x4"),
    ("vgg13_vanilla", "vgg13"),
    ("ResNet50_vanilla", "ResNet50"),
]

def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        return torch.load(path, map_location="cpu")

for directory, model_name in teachers:
    path = os.path.join(root, "save", "models", directory, "ckpt_epoch_240.pth")
    checkpoint = load_checkpoint(path)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise RuntimeError(f"{path}: unsupported checkpoint structure {type(checkpoint).__name__}")

    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}

    model = model_dict_teacher[model_name](num_classes=100)
    model.load_state_dict(state, strict=True)

    accuracy = None
    if isinstance(checkpoint, dict):
        accuracy = checkpoint.get("best_acc", checkpoint.get("accuracy"))
    suffix = f" (recorded accuracy: {accuracy})" if accuracy is not None else ""
    print(f"[ OK] {directory}: compatible{suffix}")

    del checkpoint, state, model
    gc.collect()
PY
        ) || die "Checkpoint verification failed. Rerun with --force; use --no-verify only to bypass this check intentionally."
        ok "All six checkpoints are compatible with AKD."
    else
        warn "python3 with PyTorch was not available, so compatibility verification was skipped."
    fi
fi

printf '\n'
ok "Pretrained teachers are ready."
printf 'Example AKD command:\n  cd %q\n  python train_cifar_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill AKD --model_t resnet32x4 --model_s resnet8x4\n' "$REPO_ROOT"
