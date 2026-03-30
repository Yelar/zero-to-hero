#!/usr/bin/env python3
"""Small dependency-free checks for the micrograd implementation."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from micrograd.engine import Value
from micrograd.nn import MLP


def expression(a_value: float, b_value: float) -> tuple[Value, Value, Value]:
    a = Value(a_value)
    b = Value(b_value)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    return g, a, b


def finite_difference_grad(var: str, h: float = 1e-6) -> float:
    if var == "a":
        plus = expression(-4.0 + h, 2.0)[0].data
        minus = expression(-4.0 - h, 2.0)[0].data
    elif var == "b":
        plus = expression(-4.0, 2.0 + h)[0].data
        minus = expression(-4.0, 2.0 - h)[0].data
    else:
        raise ValueError(var)
    return (plus - minus) / (2 * h)


def gradient_check() -> None:
    out, a, b = expression(-4.0, 2.0)
    out.backward()
    numeric_a = finite_difference_grad("a")
    numeric_b = finite_difference_grad("b")

    print("micrograd scalar gradient check")
    print(f"forward g.data={out.data:.6f}")
    print(f"autograd dg/da={a.grad:.6f} finite_diff={numeric_a:.6f}")
    print(f"autograd dg/db={b.grad:.6f} finite_diff={numeric_b:.6f}")

    assert math.isclose(a.grad, numeric_a, rel_tol=1e-5, abs_tol=1e-5)
    assert math.isclose(b.grad, numeric_b, rel_tol=1e-5, abs_tol=1e-5)


def toy_mlp_train() -> None:
    random.seed(0)
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    model = MLP(3, [4, 4, 1])

    losses: list[float] = []
    for _ in range(200):
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))
        losses.append(loss.data)

        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data += -0.01 * p.grad

    final_preds = [model(x).data for x in xs]
    print()
    print("micrograd toy MLP fit")
    print(f"loss_start={losses[0]:.6f} loss_end={losses[-1]:.6f}")
    print("predictions=" + ", ".join(f"{p:.3f}" for p in final_preds))

    assert losses[-1] < losses[0]


if __name__ == "__main__":
    gradient_check()
    toy_mlp_train()
    print()
    print("status=passed")
