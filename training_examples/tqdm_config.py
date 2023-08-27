import os
from nn.helpers.envvar import getenv


os.environ["TQDM_COLOUR"]="GREEN"
os.environ["TQDM_BAR_FORMAT"]="{n_fmt}/{total_fmt} Epochs | {elapsed} Elapsed | {desc} | {bar:50} |{postfix}"
os.environ["TQDM_DESC"]="Accuracy Train = {:.2%} | Accuracy Test = {:.2%}".format(0, 0)

# Disable progress bars for INFO2 as to not clutter the console.
LOGLEVEL = getenv('LOGLEVEL', 'WARNING')
if LOGLEVEL == "INFO2":
    os.environ["TQDM_DISABLE"]="True"