1. Install `poetry` if you don't have it:
```
pip install poetry
```

2. Install dependecies with poetry:
```
poetry install
```

3. Activate environment if it didn't activate automatically:
```
poetry env activate
```

4. After deploying dask scheduler and  workers as described in [documentation](https://docs.dask.org/en/stable/deploying-cli.html), you can run the client:
```
python main.py
```
Or, without activating environment:
```
poetry run python main.py
```