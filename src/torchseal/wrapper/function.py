from typing import Optional
from torch.autograd.function import NestedIOFunction
from torchseal.wrapper import CKKSWrapper

import numpy as np


class CKKSLinearFunctionWrapper(NestedIOFunction):
    __enc_input: Optional[CKKSWrapper] = None
    __weight: Optional[CKKSWrapper] = None

    @property
    def enc_input(self) -> CKKSWrapper:
        assert self.__enc_input is not None, "enc_input is not set"

        return self.__enc_input

    @enc_input.setter
    def enc_input(self, enc_input: CKKSWrapper) -> None:
        self.__enc_input = enc_input

    @property
    def weight(self) -> CKKSWrapper:
        assert self.__weight is not None, "weight is not set"

        return self.__weight

    @weight.setter
    def weight(self, weight: CKKSWrapper) -> None:
        self.__weight = weight


class CKKSPoolingFunctionWrapper(NestedIOFunction):
    pass


class CKKSLossFunctionWrapper(NestedIOFunction):
    __enc_output: Optional[CKKSWrapper] = None
    __enc_target: Optional[CKKSWrapper] = None

    @property
    def enc_output(self) -> CKKSWrapper:
        assert self.__enc_output is not None, "enc_output is not set"

        return self.__enc_output

    @enc_output.setter
    def enc_output(self, enc_output: CKKSWrapper) -> None:
        self.__enc_output = enc_output

    @property
    def enc_target(self) -> CKKSWrapper:
        assert self.__enc_target is not None, "enc_target is not set"

        return self.__enc_target

    @enc_target.setter
    def enc_target(self, enc_target: CKKSWrapper) -> None:
        self.__enc_target = enc_target


class CKKSActivationFunctionWrapper(NestedIOFunction):
    __enc_input: Optional[CKKSWrapper] = None
    __deriv_coeffs: Optional[np.ndarray] = None

    @property
    def enc_input(self) -> CKKSWrapper:
        assert self.__enc_input is not None, "enc_input is not set"

        return self.__enc_input

    @enc_input.setter
    def enc_input(self, enc_input: CKKSWrapper) -> None:
        self.__enc_input = enc_input

    @property
    def deriv_coeffs(self) -> np.ndarray:
        assert self.__deriv_coeffs is not None, "deriv_coeffs is not set"

        return self.__deriv_coeffs

    @deriv_coeffs.setter
    def deriv_coeffs(self, deriv_coeffs: np.ndarray) -> None:
        self.__deriv_coeffs = deriv_coeffs


class CKKSSoftmaxFunctionWrapper(NestedIOFunction):
    __enc_output: Optional[CKKSWrapper] = None

    @property
    def enc_output(self) -> CKKSWrapper:
        assert self.__enc_output is not None, "enc_output is not set"

        return self.__enc_output

    @enc_output.setter
    def enc_output(self, enc_output: CKKSWrapper) -> None:
        self.__enc_output = enc_output
