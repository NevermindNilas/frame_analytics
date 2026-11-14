import time, sys, torch
import frame_analytics as fa
from frame_analytics import backend, functional as Fn

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
a = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8, device=dev)
b = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8, device=dev)
af, bf = a.float() / 255, b.float() / 255


def t(fn, n=3000):
    for _ in range(200):
        fn()
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6


cases = {
    "psnr u8": lambda: fa.psnr(a, b, data_range=255.0),
    "mse f32": lambda: fa.mse(af, bf),
    "ssim u8": lambda: fa.ssim(a, b, data_range=255.0),
    "ssim f32": lambda: fa.ssim(af, bf),
    "ssim none": lambda: fa.ssim(af, bf, reduction="none"),
    "gmsd u8": lambda: fa.gmsd(a, b, data_range=255.0),
    "l1 u8": lambda: fa.l1(a, b),
    "ms_ssim(256) u8": None,
}
a2 = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8, device=dev)
b2 = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8, device=dev)
cases["ms_ssim(256) u8"] = lambda: fa.ms_ssim(a2, b2, data_range=255.0)
if dev == "cuda":
    cases["stream lookup"] = lambda: torch.cuda.current_stream(a.device).cuda_stream
    cases["raw stream"] = lambda: torch._C._cuda_getCurrentRawStream(0)
    cases["device.index"] = lambda: a.device.index
    cases["get_device"] = lambda: a.get_device()
cases["empty f64"] = lambda: torch.empty((), device=dev, dtype=torch.float64)
cases["empty f64 2d"] = lambda: torch.empty((3, 16), device=dev, dtype=torch.float64)
cases["data_ptr"] = lambda: a.data_ptr()
cases["_prep"] = lambda: Fn._prep(a, b, None, 0, None)
cases["is_grad_enabled"] = lambda: torch.is_grad_enabled()

with torch.no_grad():
    for k, fn in cases.items():
        print(f"{k:18} {t(fn):8.2f} us")
