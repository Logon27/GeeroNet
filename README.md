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

Available log levels:
- INFO
- WARN (Default)
- DEBUG

```
export LOGLEVEL=INFO
```