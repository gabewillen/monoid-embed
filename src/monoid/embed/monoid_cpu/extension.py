import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXTENSION_CACHE = {}


def load_monoid_cpu_ext(
    verbose: bool = False,
    fast_math: bool | None = None,
    fast_tanh: bool | None = None,
):
    march = os.getenv("MONOID_CPU_MARCH", "native")
    openmp = os.getenv("MONOID_CPU_OPENMP", "1")
    debug = os.getenv("MONOID_CPU_DEBUG", "0")
    if fast_math is None:
        fast_math = os.getenv("MONOID_CPU_FAST_MATH", "1") == "1"
    if fast_tanh is None:
        fast_tanh_env = os.getenv("MONOID_CPU_FAST_TANH")
        if fast_tanh_env is not None:
            fast_tanh = fast_tanh_env == "1"
        else:
            fast_tanh = bool(fast_math)

    key = (march, openmp, debug, bool(fast_math), bool(fast_tanh))
    if key in _EXTENSION_CACHE:
        return _EXTENSION_CACHE[key]

    this_dir = Path(__file__).resolve().parent
    source = str(this_dir / "monoid_cpu.cpp")
    name = "monoid_cpu_ext"
    if fast_math:
        name += "_fm"
    if fast_tanh:
        name += "_ft"
    if debug == "1":
        name += "_dbg"
    build_dir = Path(os.getenv("MONOID_CPU_BUILD_DIR", str(Path("tmp") / "torch_ext_monoid")))
    build_dir.mkdir(parents=True, exist_ok=True)

    extra_cflags = ["-O3", "-std=c++17", "-funroll-loops"]
    if march:
        extra_cflags.append(f"-march={march}")
    extra_ldflags = []
    if openmp == "1":
        extra_cflags.append("-fopenmp")
        extra_ldflags.append("-fopenmp")
    if fast_math:
        extra_cflags.append("-ffast-math")
    if fast_tanh:
        extra_cflags.append("-DMONOID_FAST_TANH=1")
    if debug == "1":
        extra_cflags.append("-g")
        extra_cflags.append("-fno-omit-frame-pointer")

    ext = load(
        name=name,
        sources=[source],
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        with_cuda=False,
        verbose=verbose,
        build_directory=str(build_dir),
    )
    _EXTENSION_CACHE[key] = ext
    return ext
