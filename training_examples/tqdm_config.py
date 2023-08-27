import os
os.environ["TQDM_COLOUR"]="GREEN"
os.environ["TQDM_BAR_FORMAT"]="{n_fmt}/{total_fmt} Epochs | {elapsed} Elapsed | {desc} | {bar:50} |{postfix}"
os.environ["TQDM_DESC"]="Accuracy Train = {:.2%} | Accuracy Test = {:.2%}".format(0, 0)

def getenv(key, default=0):
    # Gets the environment variable or the default value if the env var is not set.
    # Then casts the value to the type of 'default'
    return type(default)(os.getenv(key, default))

# Disable progress bars for INFO2 as to not clutter the console.
LOGLEVEL = getenv('LOGLEVEL', 'WARNING')
if LOGLEVEL == "INFO2":
    os.environ["TQDM_DISABLE"]="True"