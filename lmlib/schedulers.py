import math


class SchedulerBaseLR():
    '''A simple base class for learning rate schedulers'''

    def __init__(self, optimizer):
        self._optimizer = optimizer
        self.steps = 0
        self.lr = None

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        "Zero gradients with the inner optimizers"
        # TODO: see if this is necessary? can we just do it outside of this class - double check that it works
        # print("zeroing gradients here")
        self._optimizer.zero_grad()

    def _update_lr(self, lr):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def get_last_lr(self):
        # returning in list to be compatible with torch schedulers
        return [self.lr]


class InvSqrtWarmupLR(SchedulerBaseLR):
    def __init__(self, optimizer, d_model: int, warmup_steps: int, weight: int = 1):
        super().__init__(optimizer)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.weight = weight
    
    def step(self):
        # stepping first to avoid 0 lr
        self.steps += 1
        lr = self.weight * (self.d_model**(-0.5) * min(self.steps**(-0.5), self.steps * self.warmup_steps ** (-1.5)))
        self._update_lr(lr)
        self._optimizer.step() # see if I need to do the stepping here or if I am ok



class CosineWarmupLR(SchedulerBaseLR):
    def __init__(self, optimizer, warmup_steps: int, lr_decay_iters: int, min_lr: float, max_lr: float):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        print(f"CosineWarmupLR Config: {warmup_steps=}, {lr_decay_iters=}, {min_lr=}, {max_lr}")

    
    def step(self):
        # linear warmup
        if self.steps < self.warmup_steps:
            # print("warming up")
            lr =  self.max_lr * self.steps / self.warmup_steps
        # post-decay, return min_lr
        elif self.steps > self.lr_decay_iters:
            lr = self.min_lr
        # cosine decay down to min learning rate
        else:
            # print("decaying")
            decay_ratio = (self.steps - self.warmup_steps) / (self.lr_decay_iters - self.warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)

        # if self.steps % 100 == 0:
        #     print(f"Cosine warmup LR: {lr}")

        self.steps += 1
        # update lr in optimizer
        self._update_lr(lr)
        self._optimizer.step()

