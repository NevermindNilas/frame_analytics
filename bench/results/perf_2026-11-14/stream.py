import torch, time, numpy as np
import frame_analytics as fa
for metrics in [("psnr","ssim","ms_ssim","gmsd"), ("mse","psnr","ssim")]:
    sm = fa.StreamingMetrics((1,3,1080,1920), device="cuda", dtype=torch.uint8, metrics=metrics)
    a = torch.randint(0,256,(1,3,1080,1920),dtype=torch.uint8,device="cuda"); b = torch.randint(0,256,(1,3,1080,1920),dtype=torch.uint8,device="cuda")
    ah, bh = a.cpu().numpy(), b.cpu().numpy()
    r = sm.update(a, b)
    ref = {k: float(getattr(fa, k)(a, b, data_range=255.0)) if k not in ("mse",) else float(fa.mse(a,b)) for k in metrics}
    print(metrics, "graph", sm.graph_captured, "maxdiff", max(abs(r[k]-ref[k]) for k in metrics))
    for name, (x, y) in {"resident": (a, b), "host": (ah, bh)}.items():
        for _ in range(20): sm.update(x, y)
        best = 1e9
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(100): sm.update(x, y)
            best = min(best, (time.perf_counter()-t0)/100)
        print(f"  {name}: {best*1e3:.3f} ms/frame = {1/best:.0f} fps")
