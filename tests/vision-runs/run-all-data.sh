set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python}"

"$python_bin" "$here/fetch_data/class_dir_oxford_iiit_pet.py" --target-dir "$here/dataset/class-dir/oxford-iiit-pet"
