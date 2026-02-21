import torch as t

def train():
    is_fork = t.multiprocessing.get_start_method() == "fork"

    device = (
    t.device(0)
    if t.cuda.is_available and not is_fork
    else t.device("cpu")
    )




