# Environment Variables

This is a list of environment variables that control the behavior of Geero at runtime.

Example: 
```bash
DISABLE_JIT=1 python mnist.py
```

Alternatively, you can export an env var for consistent usage...

```bash
# Export the env var
export DISABLE_JIT=1

# Run your program
python mnist.py

# Unset the env var
unset DISABLE_JIT
```

## Environment Variable List

| Variable    | Possible Values | Notes |
| ----------- | --------------- | ----- |
| DISABLE_JIT | [1]             | Disable JIT for debugging purposes |
| JAX_PLATFORM_NAME | [cpu, gpu] | Choose whether to execute your code on cpu or gpu |

## Logging Levels

Example: `LOGLEVEL=INFO2 python mnist.py`

| Logging Level | Value         | Notes |
| ------------- | ------------- | ----- |
| CRITICAL      | 50            |       |
| ERROR         | 40            |       |
| WARNING       | 30            |       |
| INFO          | 20            |       |
| INFO2         | 19            | Enhanced logging for forward pass |
| DEBUG         | 10            |       |
| NOTSET        | 0             |       |

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