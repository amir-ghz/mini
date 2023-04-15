import torch


def init():
    global activation_bw
    activation_bw = 8


def quantizie(tensor, bit_width):

    # print("works")

    min = torch.min(tensor).item()
    max = torch.max(tensor).item()
    scaling_fac = (max-min)/((2**bit_width)-1)
    quantized_tensor = torch.round(tensor/scaling_fac)
    tensor = quantized_tensor*scaling_fac

    return tensor
