def invsqrt_warm(step, d_model, warmup_steps, weight):
    
    if step == 0:
        step = 1
    # print(f"{step=}, {d_model=}, {warmup_steps=}, {weight=}")
    out = weight * (d_model**(-0.5) * min(step**(-0.5), step * warmup_steps ** (-1.5)))
    # print(f"{out=}")
    return out