# Deep Learning from Scratch 3

## Environment

This repository uses "pyenv" for control python version and "poetry" for setting virtual environment like "conda".

### pyenv in Windows11

https://github.com/pyenv-win/pyenv-win#quick-start

If you want to set python version in local, Use command line "pyenv local < python version >"

```pwsh
pyenv local <python version> (3.10.10)
pyenv global <python version> (3.10.10)
``````
    

### Poetry in Windows11

https://python-poetry.org/docs/#installing-with-the-official-installer

```pwsh
poetry init
poetry config virtualenvs.in-project true --local
poetry env use python3.10
poetry run python --version  ## check version
poetry install
```
    
Install pytorch in poetry

```pwsh
poetry source add -p explicit pytorch https://download.pytorch.org/whl/cu117
poetry add --source pytorch torch torchvision
poetry run python -c "import torch;print(torch.cuda.is_available())"  ## check pytorch installed
poetry add black flake8 mypy isort --group dev
```




<br><br>
<hr>

## Learning Schedule


|Chapter List|Date|
|----|---------|
|Environment setting|2023.11.18 ~ 2023.11.19|
|Chapter 1 Traning|2023.11.20|
|Chapter 2 Traning|-|
|Chapter 3 Traning|-|
|Chapter 4 Traning|-|
|Chapter 5 Traning|-|

<hr>

