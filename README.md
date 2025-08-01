# LTN Disassembler

## How to Use?

1. You need have Ghidra installed in your system
2. Like the command in `env.sh`, an environment variable `GHIDRA_INSTALL_DIR` needs to be set to your ghidra installation path.
3. Install the dependencies by `pip install -r requirements.txt`
5. Ghidra scripts are in `./ghidra_scripts`. The most useful two are `batch_training.py` and `ltn_helper.py`. The previous one is used for training one single binary and the second is for training multiple binaries under the same directory. 
6. A CUDA device is preferred for training. 
7. `python3 ghidra_scripts/ltn_helper.py {$BINARY_FILE} debug` for running `ltn_helper.py`; `python3 ghidra_scripts/batch_training.py` for running `batch_training.py`. 
7. A `debug` folder is preferred for storing the debug information. 