import weakref
import contextlib
from typing import Union, Any

import numpy as np


class Config:
    enable_backprop = True


class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name: str = None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}는 지원하지 않습니다.')
        self.data = data
        self.grad: np.ndarray = None
        self.name = name
        self.creator: Function = None
        self.generation = 0

    def set_creator(self, func: 'Function'):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad: bool = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  ## data와 grad의 type을 맞춰줌.

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)
        while funcs:
            func = funcs.pop()
            gys = [output().grad for output in func.outputs]
            gxs = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(func.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in func.outputs:
                    y().grad = None
    
    def cleargrad(self):
        self.grad = None

    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    @property
    def size(self) -> int:
        return self.data.size
    
    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'

    def __mul__(self, other: Union['Variable', np.ndarray]):
        return mul(self, other)
    
    def __rmul__(self, other: Union['Variable', np.ndarray]):
        return mul(self, other)

    def __add__(self, other: Union['Variable', np.ndarray]):
        return add(self, other)
    
    def __radd__(self, other: Union['Variable', np.ndarray]):
        return add(self, other)
    
    def __neg__(self):
        return neg(self)
    
    def __sub__(self, other: Union['Variable', np.ndarray]):
        return sub(self, other)
    
    def __rsub__(self, other: Union['Variable', np.ndarray]):
        return rsub(self, other)

    def __truediv__(self, other: Union['Variable', np.ndarray]):
        return div(self, other)
    
    def __rtruediv__(self, other: Union['Variable', np.ndarray]):
        return rdiv(self, other)
    
    def __pow__(self, c: float):
        return pow(self, c)


class Function:
    def __call__(self, *inputs: list[Variable]):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()
    
    def backward(self, gy: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 + x1
        return y
    
    def backward(self, gy: np.ndarray) -> np.ndarray: 
        return gy, gy
    

class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx    


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1
        return y
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 - x1
        return y
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        return gy, -gy


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 / x1
        return y
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c: float):
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x ** self.c
        return y
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx

def as_array(x: Union[int, float, np.ndarray]) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj: Union[Variable, np.ndarray]) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def add(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    x1 = as_array(x1)
    return Add()(x0, x1)


def square(x: np.ndarray) -> np.ndarray:
    return Square()(x)
    

def mul(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    return Mul()(x0, x1)
    

def neg(x: np.ndarray) -> np.ndarray:
    return Neg()(x)


def sub(x0: np.ndarray, x1: Union[int, float, np.ndarray]) -> np.ndarray:
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0: np.ndarray, x1: Union[int, float, np.ndarray]) -> np.ndarray:
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    x1 = as_array(x1)
    return Div()(x1, x0)


def pow(x: np.ndarray, c: float) -> np.ndarray:
    return Pow(c)(x)


@contextlib.contextmanager
def using_config(name: str, value: bool):
    old_value = getattr(Config, name)
    setattr(Config, name, value)  ## 역전파 False 상태
    try:
        yield
    finally:
        setattr(Config, name, old_value)  ## 역전파 True 상태


def no_grad():
    return using_config('enable_backprop', False)