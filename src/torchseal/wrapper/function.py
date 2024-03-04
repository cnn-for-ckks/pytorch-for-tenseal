from torch.autograd.function import NestedIOFunction
from tenseal import CKKSVector


class CKKSFunctionCtx(NestedIOFunction):
    enc_x: CKKSVector
