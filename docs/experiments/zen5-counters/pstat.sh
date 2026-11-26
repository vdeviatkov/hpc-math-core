#!/bin/bash
# usage: pstat.sh <label> <command...>   — one perf stat line per config, CSV
EV=cycles,fp_ret_sse_avx_ops.all,ls_any_fills_from_sys.all,ls_any_fills_from_sys.local_l2,ls_any_fills_from_sys.local_ccx,ls_any_fills_from_sys.dram_io_all
label=$1; shift
out=$(taskset -c 2 perf stat -x, -e $EV "$@" 2>&1 >/dev/null)
get() { echo "$out" | awk -F, -v e="$1" '$3==e {print $1}'; }
cyc=$(get cycles); fp=$(get fp_ret_sse_avx_ops.all); all=$(get ls_any_fills_from_sys.all)
l2=$(get ls_any_fills_from_sys.local_l2); l3=$(get ls_any_fills_from_sys.local_ccx); dr=$(get ls_any_fills_from_sys.dram_io_all)
python3 -c "
fp=$fp/1000.0
print('%s,%.2f,%.3f,%.3f,%.3f,%.4f' % ('$label', $fp*1000/$cyc, $all/fp, $l2/fp, $l3/fp, $dr/fp))"
