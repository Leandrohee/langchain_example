# Commands utils

```bash
python3 -m venv venv
source venv/bin/activate
which python  
deactivate
pip install --upgrade pip    
pip freeze > requirements.txt
poetry run python main.py
```

# Libs with pip

```bash
pip install openai
pip install langchain langchain-openai
```

# Libs with poetry

```bash
poetry init
poetry add openai dotenv
poetry add langchain langchain-openai
poetry add black --group dev
poetry add pre-commit --group dev
poetry add langchain-community pypdf
poetry add langchain_chroma
```