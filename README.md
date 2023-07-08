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

### Selecting LOGLEVEL

Available log levels:
- INFO
- WARN (Default)
- DEBUG

```
export LOGLEVEL=INFO
```