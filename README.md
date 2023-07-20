# GeeroNet

## Installing Google Jax
```
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

## Installing Pytorch (Used For Data Loading)
```
pip install torch torchvision
```

## Setting Up A Venv
```
python -m venv venvGeeroNet

# Linux (WSL2)
. venvGeeroNet/bin/activate

source venvGeeroNet/bin/activate

# Windows
venvGeeroNet/scripts/activate.bat

pip install -r requirements.txt
```

If you are having trouble getting the venv to activate when it has worked properly in the past...
```
CTRL + SHIFT + P -> Search 'Reload' -> Click 'Python: Clear Cache and Reload Window'
```

### Selecting LOGLEVEL

| Logging Level | Value | Notes |
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
```

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