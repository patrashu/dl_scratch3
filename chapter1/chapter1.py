import unittest
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

    def set_creator(self, func: 'Function'):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  ## data와 grad의 type을 맞춰줌.

        funcs = [self.creator]
        while funcs:
            func = funcs.pop()
            x, y = func.input, func.output
            x.grad = func.backward(y.grad)
            
            if x.creator is not None:
                funcs.append(x.creator)  


class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(asarray(y))
        output.set_creator(self)
        self.input = input  ## input data 유지
        self.output = output  ## output 저장
        return output
    
    ## 하위 클래스에서 반드시 구현해야 함.
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()  
    
    ## 미분 계산
    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)
    
    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


# 입력이 np.ndarray를 보장
def asarray(x: Union[int, float, np.ndarray]) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


#########################################################################
## Test Code

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


A = Square()
B = Exp()
C = Square()


x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad=np.array(1.0)
y.backward()
print(x.grad)
