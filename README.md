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

Sometimes VSCode will automatically activate the venv, sometimes it does not. You can use one of the above commands to activate the venv. Or sometimes just closing and reopening your terminal will cause it to properly source your venv.

#### Troubleshooting venv activation

If you are having trouble getting the venv to activate (at all) when it has worked properly in the past...
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

### GPU Setup

By default the requirements.txt file will install jax for cpu. However, to take advantage of the gpu you must install both cuda and jax dependencies. The easiest way to install these dependencies is through pip. Instructions can be found in the [README](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier) of the [jax repo](https://github.com/google/jax). Once these dependencies are installed, the gpu should automatically be used when using any Jax or Geero functionality.

---

## Environment Variables And Logging Levels

For a list of available environment variables and logging levels, please see [env_vars](env_vars.md).

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

```python
input_shape = (input_size,)
# or...
input_shape = (-1, input_size)
# or if you have a fixed batch size (such as 128)... 
input_shape = (128, input_size)
```

These variations only really affect the output of some of the debug shape print statements. The more information you give Geero, the more accurate shapes it will display in the debug statements.

#### Conv
The initialization (init_fun) of the Conv layer is not dependent on the batch size. It can also be independent of the input_height and input_width, but only if you have a purely convolutional network. This is because if you have Dense layers mixed in, the Dense layers need to know the flattened output shape of the Conv layer. This is because unlike Conv, the Dense layer IS dependent on the input size.  

```python
input_shape = (-1, input_height, input_width, input_channels)

# ONLY if your network is fully convolutional can you do... 
input_shape = (-1, -1, -1, input_channels)
# Although this is not recommended if you DO have a fixed input size.
# Because the debugging information will be less useful.
```