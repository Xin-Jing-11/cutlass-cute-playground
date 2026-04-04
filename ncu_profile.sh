# ncu_profile.sh — Run NVIDIA Nsight Compute on a Python script
    # Usage: ./ncu_profile.sh bench_sgemm.py [extra args...]
    #   e.g. ./ncu_profile.sh bench_sgemm.py --method cutlass

    set -euo pipefail

    SCRIPT="${1:?Usage: $0 <python_script> [args...]}"
    shift

    ncu \
      --set full \
      --target-processes all \
      -o "ncu_report/ncu_$(basename "${SCRIPT}" .py)" \
      python3 "$SCRIPT" "$@"

    echo "Report saved to ncu_$(basename "${SCRIPT}" .py).ncu-rep"