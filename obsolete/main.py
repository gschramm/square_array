from array_api_compat import device, array_namespace
from utils import square_array_backend


def square_and_sum(xp,n = 10, dev = None):
    arr = xp.arange(1, n+1, dtype=xp.float32, device=dev)

    print("")
    print("---------------------------------")
    print("---------------------------------")
    print(array_namespace(arr))
    print(device(arr))
    print("Before:", arr)
    
    sum_sq = square_array_backend(arr)
    
    print("After: ", arr)
    print("Sum of squares:", sum_sq)


if __name__ == "__main__":

    import array_api_compat.numpy as np
    square_and_sum(np)
  
    try:
        import array_api_strict as nparr
        square_and_sum(nparr)
    except:
        print("\narray api strict")

    try:
        import array_api_compat.cupy as cp
        square_and_sum(cp)
    except:
        print("\nskipping cupy")

    try:
        import array_api_compat.torch as torch
        square_and_sum(torch, dev = "cpu")
    except:
        print("\nskipping torch cpu")

    try:
        import array_api_compat.torch as torch
        square_and_sum(torch, dev = "cuda:0")
    except:
        print("\nskipping torch cuda")
