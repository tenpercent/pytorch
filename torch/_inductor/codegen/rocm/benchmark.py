import logging
import os
import sys

import pandas  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

import torch
from torch._dynamo import config as dynconfig
from torch._inductor import config
from torch._inductor.codegen.rocm.ck_template import CKTemplate

log = logging.getLogger(__name__)


def generate_inputs(M, N, K, Bias, tensor_options, layout, f=torch.randn):
    if layout[0] == "r":
        a = f(M, K, **tensor_options)
    elif layout[0] == "c":
        a = f(K, M, **tensor_options).transpose(0, 1)
    else:
        a = None

    if layout[1] == "r":
        b = f(K, N, **tensor_options)
    elif layout[1] == "c":
        b = f(N, K, **tensor_options).transpose(0, 1)
    else:
        b = None

    if layout[2] == "r":
        out = torch.empty(M, N, **tensor_options)
    elif layout[2] == "c":
        out = torch.empty(N, M, **tensor_options).transpose(0, 1)
    else:
        out = None

    if Bias:
        bias = f(Bias, **tensor_options)
    else:
        bias = None

    return a, b, bias, out


def main(gemm_shape_csv, layout, dtype):
    import ck4inductor

    ck_dir = os.path.dirname(ck4inductor.__file__)
    os.environ["TORCHINDUCTOR_CK_DIR"] = ck_dir

    df = pandas.read_csv(gemm_shape_csv, dtype=pandas.Int64Dtype())

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    tensor_options = {"device": "cuda", "dtype": dtype}

    problem_instances = df[["M", "K", "N", "Bias"]].values

    def mm(a, b, out):
        return torch.mm(a, b, out=out)

    def addmm(a, b, bias, out):
        return torch.addmm(bias, a, b, out=out)

    for M, K, N, Bias in (pbar := tqdm(problem_instances)):
        if Bias is pandas.NA:
            Bias = None
        pbar.set_description(f"{M=} {N=} {K=} {Bias=}")
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CK,Triton,ATen",
                "compile_threads": 64,
                "trace.enabled": True,
                "trace.log_autotuning_results": True,
                "rocm.ck_dir": ck_dir
            }
        ), dynconfig.patch(
            {"cache_size_limit": len(problem_instances) + 1}
        ), torch.no_grad():
            a, b, bias, out = generate_inputs(M, N, K, Bias, tensor_options, layout)
            if Bias:
                Y_compiled = torch.compile(addmm, dynamic=False)(a, b, bias, out)
                Y = torch.addmm(bias, a, b, out=out)
            else:
                Y_compiled = torch.compile(mm, dynamic=False)(a, b, out)
                Y = mm(a, b, out)
            try:
                torch.testing.assert_close(Y_compiled, Y)
            except AssertionError as e:
                log.error(e)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].lower() == "torchbench":
        gemm_shape_csv = "https://raw.githubusercontent.com/pytorch/benchmark/main/torchbenchmark/operators/gemm/amd.csv"
    else:
        gemm_shape_csv = sys.argv[1]

    if len(sys.argv) < 3:
        layout = "rcr"
    else:
        layout = sys.argv[2].lower()

    if len(sys.argv) < 4:
        dtype = torch.half
    else:
        # as long as the mapping is 1:1 this is fine
        ck_dtype_to_torch = {v: k for k, v in CKTemplate._TORCH_DTYPE_TO_CK.items()}
        dtype = ck_dtype_to_torch[sys.argv[3].upper()]
    main(gemm_shape_csv, layout, dtype)
