# Dafne Dataset

## Install
You can install the package from the repository using `pip` or your preferred package manager (usgin `uv` is suggested). 

### User mode

To install and import the package in your project run
```bash
# standard pip command
pip install git+https://github.com/emarj/DAFNE_dataset.git
# uv in project mode
uv add git+https://github.com/emarj/DAFNE_dataset.git
# uv in venv only mode
uv pip install git+https://github.com/emarj/DAFNE_dataset.git 
```

### Dev mode
If you want to play with it, checkout the repo
```bash
git clone https://github.com/emarj/DAFNE_dataset.git
```

```bash
uv sync
```
this will automatically create a `venv` for you. Otherwise
```bash
# create your venv
pip install .
```
