# Autodiff + MLP From Scratch: A Full Educational Tutorial

## Who this is for
This is for you if you:
- can write basic Python
- know what a derivative is
- can use neural nets in frameworks, but still feel that backprop is "magic"

By the end, you will be able to implement from scratch:
- scalar reverse-mode autodiff (micrograd style)
- a tiny neural-net library (`Neuron`, `Layer`, `MLP`)
- training with SGD and Adam
- debugging tools (finite differences, sanity checks, failure diagnosis)

## How to use this tutorial
Use this in 3 passes:
1. Read + run every block once, top to bottom.
2. Re-read and pause before each code block to predict results.
3. Rebuild in a blank file from memory.

You only really internalize this when pass 3 works.

---

## 1) Why this matters
Frameworks hide 3 core things from you:
1. How gradients are created.
2. Why gradients are accumulated.
3. Where training bugs come from.

When you build the engine yourself, you get:
- mechanical understanding of the chain rule
- instinct for debugging exploding/zero gradients
- confidence to inspect model behavior instead of guessing

---

## 2) Backprop from first principles

### 2.1 Local derivatives
Suppose:

$$
f(x, y, z) = x y + z
$$

Then:
- $\partial f/\partial x = y$
- $\partial f/\partial y = x$
- $\partial f/\partial z = 1$

Backprop is just repeated application of this idea on a graph.

### 2.2 Graph view
Think in nodes and edges:
- each node stores a scalar value
- each edge contributes a local derivative
- gradients flow from output backward to inputs

### 2.3 Chain rule in graph form
For any node `u` and downstream node `v`:

$$
\frac{\partial L}{\partial u} += \frac{\partial L}{\partial v} \cdot \frac{\partial v}{\partial u}
$$

That `+=` is critical. If one node influences output through multiple paths, gradients add.

---

## 3) Build a scalar autodiff engine

### 3.1 The `Value` object
Each scalar must store:
- `data`: forward value
- `grad`: d(output)/d(this node)
- `_prev`: parent nodes in computation graph
- `_backward`: local function that pushes gradients to parents

### 3.2 Operator rules
For each operation we need forward + local backward:

- `out = a + b`
  - forward: `out.data = a.data + b.data`
  - backward: `a.grad += out.grad`, `b.grad += out.grad`

- `out = a * b`
  - backward: `a.grad += b.data * out.grad`, `b.grad += a.data * out.grad`

- `out = a ** k`
  - backward: `a.grad += k * a.data**(k-1) * out.grad`

- `out = relu(a)`
  - backward: `a.grad += (out.data > 0) * out.grad`

### 3.3 Reverse pass with topological order
You cannot backprop in arbitrary order. You need a topological ordering of the DAG.

Steps:
1. DFS from output to collect nodes in topological order.
2. Set output gradient to 1.
3. Traverse reversed topological order and call each node's `_backward`.

### 3.4 Full engine code

```python
class Value:
    """A scalar value with reverse-mode autodiff support."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers are supported"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1.0)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # Convenience ops
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)
```

---

## 4) Verify the engine correctly

Never trust autodiff code without checks.

### 4.1 Deterministic sanity tests

```python
def assert_close(a, b, tol=1e-6, name=''):
    if abs(a - b) > tol:
        raise AssertionError(f"{name}: expected {b}, got {a}")

# shared-node accumulation: y = x*x + x => dy/dx = 2x + 1
x = Value(3.0)
y = x * x + x
y.backward()
assert_close(y.data, 12.0, name='forward y')
assert_close(x.grad, 7.0, name='dy/dx')
print('shared-node test passed')
```

### 4.2 Classic micrograd smoke test

```python
a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b ** 3
c = c + c + 1
c = c + 1 + c + (-a)
d = d + d * 2 + (b + a).relu()
d = d + 3 * d + (b - a).relu()
e = c - d
f = e ** 2
g = f / 2.0
g = g + 10.0 / f

g.backward()
print(g.data, a.grad, b.grad)
# expected ~24.7041, ~138.8338, ~645.5773
```

### 4.3 Finite differences (gold standard debug tool)

```python
def scalar_program_float(a, b):
    relu_arg = a * b + a + 2.0
    relu_val = relu_arg if relu_arg > 0 else 0.0
    return relu_val * (b ** 2) + a / (b - 0.5)

def scalar_program_value(a, b):
    av, bv = Value(a), Value(b)
    y = ((av * bv + av + 2.0).relu() * (bv ** 2)) + av / (bv - 0.5)
    return y, av, bv

def finite_diff_2var(f, a, b, wrt='a', eps=1e-6):
    if wrt == 'a':
        return (f(a + eps, b) - f(a - eps, b)) / (2 * eps)
    return (f(a, b + eps) - f(a, b - eps)) / (2 * eps)

y, a, b = scalar_program_value(-1.5, 2.0)
y.backward()
num_da = finite_diff_2var(scalar_program_float, -1.5, 2.0, wrt='a')
num_db = finite_diff_2var(scalar_program_float, -1.5, 2.0, wrt='b')

print('autograd da/db:', a.grad, b.grad)
print('numeric  da/db:', num_da, num_db)
```

If these differ materially, your local derivative code is wrong.

---

## 5) Build MLP components on top

### 5.1 Abstractions
- `Module`: common base with `parameters()` and `zero_grad()`
- `Neuron`: weighted sum + optional ReLU
- `Layer`: list of neurons
- `MLP`: stack of layers

### 5.2 Code

```python
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        sizes = [nin] + list(nouts)
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

### 5.3 Parameter-count sanity check
For `MLP(3, [4, 4, 1])`:
- Layer 1: `3*4 + 4 = 16`
- Layer 2: `4*4 + 4 = 20`
- Layer 3: `4*1 + 1 = 5`
- Total: `41`

If your count differs, your model structure is wrong.

---

## 6) Define the learning problem

Use tiny binary dataset with labels in `{+1, -1}`.

### 6.1 Data and margin loss

```python
X = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
y = [1.0, -1.0, -1.0, 1.0]

def margin_loss(model, xs, ys, alpha=1e-4):
    scores = [model([Value(v) for v in x]) for x in xs]
    losses = [(1.0 + (-yi) * score).relu() for yi, score in zip(ys, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total = data_loss + reg_loss

    acc = sum((yi > 0) == (score.data > 0) for yi, score in zip(ys, scores)) / len(ys)
    return total, acc
```

### 6.2 Why this loss
`relu(1 - y*score)` enforces margin:
- if point is correctly classified with enough margin, loss is 0
- otherwise, loss grows linearly

Good for understanding because the derivative behavior is explicit and sparse.

---

## 7) Optimizers: SGD and Adam

### 7.1 SGD

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

Simple, strong baseline.

### 7.2 Adam intuition
Tracks moving averages of gradient and squared gradient:
- first moment: direction
- second moment: scale

Update:
- $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
- $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
- bias-correct to get `m_hat`, `v_hat`
- apply scaled step

### 7.3 Code

```python
import math

class SGD:
    def __init__(self, params, lr=0.05):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

class Adam:
    def __init__(self, params, lr=0.03, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [0.0 for _ in self.params]
        self.v = [0.0 for _ in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)

            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
```

---

## 8) Training loop

```python
def grad_l2_norm(params):
    return math.sqrt(sum((p.grad * p.grad for p in params)))

def train(model, optimizer, xs, ys, steps=120, report_every=20, zero_each_step=True):
    hist = {'loss': [], 'acc': [], 'grad_norm': []}

    for step in range(steps):
        total, acc = margin_loss(model, xs, ys)

        if zero_each_step:
            model.zero_grad()

        total.backward()
        hist['grad_norm'].append(grad_l2_norm(model.parameters()))
        optimizer.step()

        hist['loss'].append(total.data)
        hist['acc'].append(acc)

        if step % report_every == 0 or step == steps - 1:
            print(f"step={step:03d} loss={total.data:.4f} acc={acc:.2f} grad_norm={hist['grad_norm'][-1]:.4f}")

    return hist

SEED = 1337
random.seed(SEED)
model_sgd = MLP(3, [4, 4, 1])
random.seed(SEED)
model_adam = MLP(3, [4, 4, 1])

hist_sgd = train(model_sgd, SGD(model_sgd.parameters(), lr=0.05), X, y)
hist_adam = train(model_adam, Adam(model_adam.parameters(), lr=0.03), X, y)
```

---

## 9) Visualization for intuition

```python
import matplotlib.pyplot as plt

steps = list(range(len(hist_sgd['loss'])))
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(steps, hist_sgd['loss'], label='SGD')
axes[0].plot(steps, hist_adam['loss'], label='Adam')
axes[0].set_title('Loss vs Step')
axes[0].legend(); axes[0].grid(alpha=0.25)

axes[1].plot(steps, hist_sgd['acc'], label='SGD')
axes[1].plot(steps, hist_adam['acc'], label='Adam')
axes[1].set_title('Accuracy vs Step')
axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(alpha=0.25)

axes[2].plot(steps, hist_sgd['grad_norm'], label='SGD')
axes[2].plot(steps, hist_adam['grad_norm'], label='Adam')
axes[2].set_title('Gradient L2 Norm')
axes[2].grid(alpha=0.25)

plt.tight_layout()
plt.show()
```

Interpretation:
- loss should trend down
- accuracy should trend up
- grad norm should not blow up uncontrollably

---

## 10) Debugging playbook (most important practical section)

### Symptom: loss never changes
Check:
1. Did you call `backward()`?
2. Did you call `optimizer.step()`?
3. Are model parameters actually in `parameters()`?
4. Are labels/signs correct in your loss?

### Symptom: training works for a few steps then degrades
Check:
1. Did you forget `zero_grad()`?
2. Are you reusing graph nodes across iterations?

### Symptom: gradients look wrong for one variable
Check:
1. isolate one tiny expression
2. run finite-difference check on that expression
3. compare each local derivative rule

### Symptom: exploding updates
Check:
1. lower learning rate
2. inspect gradient norms
3. use Adam

### Symptom: model predicts only one class
Check:
1. initialization range
2. output activation/loss mismatch
3. label encoding mismatch

---

## 11) Common conceptual pitfalls

1. "Backprop computes derivatives by symbolic algebra"
No. It is numeric graph execution with local derivative closures.

2. "Each node has one gradient source"
No. Multiple downstream paths contribute; gradients are accumulated.

3. "ReLU derivative at 0 is always a bug"
At exactly 0 it is not uniquely defined. Choose a convention and avoid testing exactly at the kink with finite differences.

4. "If accuracy is high, implementation must be correct"
Not always. Bugs can still exist and happen to work on tiny data.

---

## 12) A mastery protocol (do this exactly)

### Stage A: guided
- run this tutorial once
- print intermediate values for 1 expression

### Stage B: memory reconstruction
- blank file
- write `Value`, operations, `backward`, tests
- no peeking until done

### Stage C: stress testing
- run 5 random finite-difference tests
- deliberately break one derivative and watch tests fail

### Stage D: extension
- add `tanh`
- add momentum SGD
- add mini-batch sampling

If you can pass Stage C quickly, you no longer treat gradients as magic.

---

## 13) Exercises (with expected outcomes)

1. Implement `tanh` in `Value`.
Expected gradient: `d(tanh(x))/dx = 1 - tanh(x)^2`.

2. Implement momentum SGD.
Expected: often smoother convergence than plain SGD.

3. Replace hinge loss with MSE on a tiny regression task.
Expected: forward/backward still works unchanged.

4. Write a generic `gradcheck(fn, params)` utility.
Expected: catches local derivative bugs in seconds.

5. Add `leaky_relu` and compare training behavior.
Expected: fewer dead activations in some runs.

---

## 14) Complete runnable script (single file)

If you want a single script version instead of notebook cells, use your existing notebook file and merge the code blocks above in this order:
1. `Value`
2. tests + finite differences
3. `Module/Neuron/Layer/MLP`
4. data + loss
5. optimizers
6. training + plots

That ordering ensures all dependencies exist when each block runs.

---

## 15) Where this maps to your repo

Reference implementations in this repo:
- `micrograd/engine.py`
- `micrograd/nn.py`
- `demo.ipynb`

Your expanded learning notebook:
- `output/jupyter-notebook/autodiff-mlp-from-scratch.ipynb`

This article:
- `output/jupyter-notebook/autodiff-mlp-full-tutorial-article.md`

---

## Final reminder
Backprop is not magic.
It is:
1. build graph during forward
2. attach local derivative rule per op
3. reverse traverse graph with chain rule and accumulation

Do this enough times in raw Python and gradients become intuitive.
