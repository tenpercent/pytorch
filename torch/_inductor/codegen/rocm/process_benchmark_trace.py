import json
import os
import re
import sys
from operator import itemgetter


def main(traces_dir):
    result_basename = "autotuning_result_json_list.txt"
    trace_dirs = []
    for root, _, files in os.walk(traces_dir):
        for f in files:
            if f == result_basename:
                trace_dirs.append(root)
    # individual run dirs have the format 'model__<N>_inference_<N>.<N>'
    sorted_dirs = sorted(trace_dirs, key=lambda f: int(re.findall(r"\d+", f)[-1]))

    trace_files = [os.path.join(r, result_basename) for r in sorted_dirs]
    display_backends = {"CK": "ROCm", "Triton": "Triton", "ATen": "extern"}
    out_lines = []
    csv_sep = ","
    out_lines.append(f"{csv_sep} ".join(display_backends.keys()))
    for filename in trace_files:
        shape_result = {}
        with open(filename, "rt") as f:
            jsons = [json.loads(line) for line in f.readlines() if line]
            key = itemgetter("benchmark_result")
            sorted_jsons = sorted(jsons, key=key)
            for j in sorted_jsons:
                for k, v in display_backends.items():
                    if j["backend"] == v:
                        if not shape_result.get(k):
                            shape_result[k] = round(key(j), 5)

        out_lines.append(
            f"{csv_sep} ".join(
                f"{shape_result.get(k, float('+inf'))} ms"
                for k in display_backends.keys()
            )
        )
    out_lines.append("")

    with open(os.path.join(traces_dir, "combined_trace.csv"), "wt") as f:
        f.write("\n".join(out_lines))


if __name__ == "__main__":
    main(sys.argv[1])
