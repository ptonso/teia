set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python}"

"$python_bin" "$here/fetch_data/boards.py" --target-dir "$here/dataset/boards"
"$python_bin" "$here/fetch_data/boards.py" --target-dir "$here/dataset/boards-graph-signal" --graph-signal
