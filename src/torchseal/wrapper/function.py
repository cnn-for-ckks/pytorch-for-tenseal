from torch.autograd.function import NestedIOFunction
from torchseal.wrapper.ckks import CKKSWrapper


class CKKSFunctionWrapper(NestedIOFunction):
    enc_x: CKKSWrapper
