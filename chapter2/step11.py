from typing import Union

import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}는 지원하지 않습니다.')
        self.data = data
        self.grad: np.ndarray = None
        self.creator: Function = None
        self.generation = 0

    def set_creator(self, func: 'Function'):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
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
            gys = [output.grad for output in func.outputs]
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
    
    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs: list[Variable]):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()
    
    def backward(self, gy: list[np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError()


def as_array(x: Union[int, float, np.ndarray]) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


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


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def add(x0, x1):
    return Add()(x0, x1)


def square(x):
    return Square()(x)
    

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)