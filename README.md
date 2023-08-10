# GeeroNet

GeeroNet is a simple neural network library built using [Google Jax](https://github.com/google/jax). This library is mainly for educational purposes and to experiment with deep learning methods.

## Setup and Installation

### Setting up a venv

#### Create a virtual environment

```bash
python -m venv venvGeeroNet
```

#### Activate your virtual environment
```bash
# Linux (WSL2)
. venvGeeroNet/bin/activate

# Alternate Linux Method
source venvGeeroNet/bin/activate
```

#### Troubleshooting venv activation

If you are having trouble getting the venv to activate when it has worked properly in the past...
```
CTRL + SHIFT + P -> Search 'Reload' -> Click 'Python: Clear Cache and Reload Window'
```

### Installing dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### WSL2 GUI Setup (required for graphics)

If using WSL2 on Windows. Please install python3-tk. This will give you a backend for GUIs. Otherwise matplotlib and similar features will not work.

```bash
sudo apt-get install python3-tk
```

---

## Geero Configuration

### Selecting LOGLEVEL

| Logging Level | Value         | Notes |
| ------------- | ------------- | ----- |
| CRITICAL      | 50            |       |
| ERROR         | 40            |       |
| WARNING       | 30            |       |
| INFO          | 20            |       |
| INFO2         | 19            | Enhanced logging for forward pass |
| DEBUG         | 10            |       |
| NOTSET        | 0             |       |

```bash
export LOGLEVEL=INFO

# To unset environment variable
unset LOGLEVEL

# Or...
LOGLEVEL=INFO python mnist.py
```

### Environment Variables

This is a list of environment variables that control the behavior of Geero at runtime.

Example: `DISABLE_JIT=1 python mnist.py`

| Variable    | Possible Values | Notes |
| ----------- | --------------- | ----- |
| DISABLE_JIT | [1]             | Disable JIT for debugging purposes |

### Breakpoint Debugging (INFO2)

For some logging levels, breakpoints are set within the code. This is to aid in debugging and make the logs readable. Using a breakpoint in jax will automatically open a jdb type prompt. Here is a list of helpful commands for interacting with the prompt. For additional information, see [this jax page](https://jax.readthedocs.io/en/latest/debugging/print_breakpoint.html#interactive-inspection-with-jax-debug-breakpoint).

- **help** - prints out available commands
- **p** - evaluates an expression and prints its result
- **pp** - evaluates an expression and pretty-prints its result
- **u(p)** - go up a stack frame
- **d(own)** - go down a stack frame
- **w(here)/bt** - print out a backtrace
- **l(ist)** - print out code context
- **c(ont(inue))** - resumes the execution of the program
- **q(uit)/exit** - exits the program (does not work on TPU)

## Docstring Format (Google)

This project utilizes the Google docstring format. It is recommended that this is configured with the autoDocstring VSCode extension.

Examples of the docstring format can be found below:
- [Basic Docstring Format](https://github.com/NilsJPWerner/autoDocstring/blob/f7bc9f427d5ebcd87e6f5839077a87ecd1cbb404/docs/google.md)
- [Detailed Docstring Examples](https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e)
- [Google Style Guide](https://google.github.io/styleguide/pyguide.html)

## Jax Utilization Within Geero

### Initializers

Currently all initializers used within GeeroNet are from the [initializers](https://jax.readthedocs.io/en/latest/jax.nn.initializers.html) implementation in Jax. I may port them over at some point for visibility. But for now you can reference and use the Jax documentation for initializers.

## Geero Rules

### Input Shapes

The -1 for the input_shape tuple is a wildcard for the batch size. The wildcard was added because the batch size can vary between forward passes after initialization. And the actual initialization of the layers is in no way dependent on the actual batch size.

#### Dense
input_shape = (input_size,)
or...
input_shape = (-1, input_size)

#### Conv
The initialization (init_fun) of the Conv layer is not dependent on the batch size. It can also be independent of the input_height and input_width, but only if you have a purely convolutional network. This is because if you have Dense layers mixed in, the Dense layers need to know the flattened output shape of the Conv layer. This is because unlike Conv, the Dense layer IS dependent on the input size.

input_shape = (-1, input_height, input_width, input_channels)

ONLY if your network is fully convolutional can you do...
input_shape = (-1, -1, -1, input_channels)