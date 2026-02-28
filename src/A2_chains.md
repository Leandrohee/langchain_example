# Documentação - chains.py

## 📋 Visão Geral
Este arquivo implementa uma aplicação LLM (Large Language Model) que recomenda cidades, restaurantes e atividades culturais com base em interesses do usuário, utilizando o LangChain como framework de orquestração.

---

## 📦 Imports e Suas Funções

### `from dotenv import load_dotenv`
- **Função**: Carrega variáveis de ambiente do arquivo `.env`
- **Uso**: Permite acessar a chave da API OpenAI de forma segura sem expô-la no código

### `from langchain_openai import ChatOpenAI`
- **Função**: Importa a classe que integra o modelo GPT da OpenAI com LangChain
- **Uso**: Permite usar o ChatGPT como o modelo de linguagem da aplicação

### `from langchain_core.output_parsers import JsonOutputParser, StrOutputParser`
- **Função**: Importa parsers que convertem as respostas do modelo em formatos específicos
  - `JsonOutputParser`: Converte resposta em JSON estruturado
  - `StrOutputParser`: Retorna a resposta como string simples
- **Uso**: Garantir que as respostas do modelo estejam no formato esperado

### `from langchain_core.prompts import PromptTemplate`
- **Função**: Classe para criar templates de prompts reutilizáveis com variáveis dinâmicas
- **Uso**: Definir prompts que aceitam diferentes entradas (ex: interesse, cidade)

### `from langchain_core.globals import set_debug`
- **Função**: Ativa o modo de debug do LangChain para rastreamento detalhado
- **Uso**: Mostrar logs detalhados da execução das chains para fins de debugging

### `import os, time`
- **Função**: 
  - `os`: Permite acessar variáveis de ambiente
  - `time`: Permite medir o tempo de execução
- **Uso**: Obter a chave da API e calcular tempo total de execução

### `from pydantic import Field, BaseModel, SecretStr`
- **Função**: Importa classes para validação e tipagem de dados
  - `BaseModel`: Classe base para criar modelos de dados estruturados
  - `Field`: Define campos com descrições e validações
  - `SecretStr`: Tipo especial para dados sensíveis (não exibe em strings)
- **Uso**: Criar estruturas de dados validadas para as respostas (cidades, restaurantes)

---

## 🔧 Seções do Código

### 1. **Configuração Inicial**
```python
set_debug(True)
start_time = time.time()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```
- **set_debug(True)**: Ativa logs detalhados do LangChain
- **start_time = time.time()**: Marca o início do programa para cronometrar
- **load_dotenv()**: Carrega variáveis do arquivo `.env`
- **api_key**: Obtém a chave da API OpenAI do ambiente

---

### 2. **Definição de Classes de Dados (Pydantic Models)**

#### Classe `Destiny`
```python
class Destiny(BaseModel):
    city: str = Field("The city recommended to visit")
    motive: str = Field("The motive for which is interesting to visit this city")
```
- **Função**: Estrutura a resposta de recomendação de cidade
- **Campos**:
  - `city`: Nome da cidade recomendada
  - `motive`: Motivo pelo qual é interessante visitar essa cidade
- **Uso**: Garantir que o modelo retorne dados estruturados e validados

#### Classe `Restaurant`
```python
class Restaurant(BaseModel):
    city: str = Field("The city recommended to visit")
    restaurants: str = Field("The restaurants in this city")
```
- **Função**: Estrutura a resposta de recomendação de restaurantes
- **Campos**:
  - `city`: Cidade onde os restaurantes estão
  - `restaurants`: Lista/descrição de restaurantes recomendados
- **Uso**: Validar a resposta sobre restaurantes

---

### 3. **Parsers de Saída**

```python
cityParser = JsonOutputParser(pydantic_object=Destiny)
restaurantParser = JsonOutputParser(pydantic_object=Restaurant)
```
- **Função**: Convertem as respostas do modelo em objetos estruturados JSON
- **cityParser**: Parseia resposta para objeto `Destiny`
- **restaurantParser**: Parseia resposta para objeto `Restaurant`
- **Uso**: Garantir que as saídas do modelo estejam no formato correto e validado

---

### 4. **Templates de Prompts**

#### `promptCity`
```python
promptCity = PromptTemplate(
    template="""
    Sugest a city knowing my interest for {interest}
    {exit_format}
    """,
    input_variables=["interest"],
    partial_variables={"exit_format": cityParser.get_format_instructions()}
)
```
- **Função**: Cria um prompt dinâmico para recomendar cidades
- **input_variables**: `["interest"]` - o usuário fornece seu interesse (ex: "beaches")
- **partial_variables**: Adiciona automaticamente instruções de formato JSON do parser
- **Resultado**: Prompt que pede recomendações e espera resposta em JSON

#### `promptRestaurant`
```python
promptRestaurant = PromptTemplate(
    template="""
    Sugest cools restaurants in this city: {city}
    {exit_format}
    """,
    input_variables=["city"],
    partial_variables={"exit_format": restaurantParser.get_format_instructions()}
)
```
- **Função**: Cria um prompt para recomendar restaurantes em uma cidade
- **input_variables**: `["city"]` - recebe o nome da cidade da chain anterior
- **Resultado**: Prompt que pede restaurantes específicos em JSON

#### `promptCultural`
```python
promptCultural = PromptTemplate(
    template="""
    Sugest nice activities and cultural places in this city: {city}
    """,
    input_variables=["city"],
)
```
- **Função**: Cria um prompt para atividades culturais
- **input_variables**: `["city"]` - recebe a cidade de orquestração anterior
- **Nota**: Este não usa um parser estruturado, retorna texto livre

---

### 5. **Configuração do Modelo**

```python
model = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0.5,
    api_key=SecretStr(api_key) if api_key else None
)
```
- **model='gpt-5-nano'**: Especifica qual modelo GPT usar
- **temperature=0.5**: Controla a criatividade
  - 0 = respostas determinísticas
  - 0.5 = equilíbrio entre criatividade e consistência
  - 1.0 = máxima criatividade
- **api_key**: Fornece a chave de autenticação da OpenAI de forma segura

---

### 6. **Definição das Chains (Pipelines de Processamento)**

```python
city_chain = promptCity | model | cityParser
restaurant_chain = promptRestaurant | model | restaurantParser
cultural_chain = promptCultural | model | StrOutputParser()
main_chain = (city_chain | restaurant_chain | cultural_chain)
```

#### **City Chain** (`promptCity | model | cityParser`)
1. Recebe o interesse do usuário
2. Passa pelo prompt que pede uma recomendação de cidade
3. Envia para o modelo GPT processar
4. Parser converte resposta em JSON validado (objeto `Destiny`)
5. **Saída**: Objeto com cidade e motivo

#### **Restaurant Chain** (`promptRestaurant | model | restaurantParser`)
1. Recebe a cidade da chain anterior
2. Passa pelo prompt que pede restaurantes
3. Envia para o modelo processar
4. Parser converte resposta em JSON validado (objeto `Restaurant`)
5. **Saída**: Objeto com restaurantes da cidade

#### **Cultural Chain** (`promptCultural | model | StrOutputParser()`)
1. Recebe a cidade
2. Passa pelo prompt que pede atividades culturais
3. Envia para o modelo processar
4. Parser retorna a resposta como string simples (sem validação rígida)
5. **Saída**: String com atividades culturais

#### **Main Chain** (Orquestração)
```python
main_chain = (city_chain | restaurant_chain | cultural_chain)
```
- **Função**: Encadeia as três chains sequencialmente
- **Fluxo**:
  1. Recomenda cidade baseada no interesse
  2. Usa essa cidade para recomendar restaurantes
  3. Usa a mesma cidade para sugerir atividades culturais
- **Resultado**: Resposta completa com cidade, restaurantes e atividades

---

### 7. **Execução e Resultado**

```python
answer = main_chain.invoke(
    {
        "interest": "beaches"
    }
)
```
- **Função**: Executa o pipeline completo
- **Entrada**: Dicionário com o interesse do usuário ("beaches")
- **Processo**: Passa por todas as 3 chains sequencialmente
- **Saída**: Resultado com cidades, restaurantes e atividades

---

### 8. **Impressão de Resultados e Tempo**

```python
end_time = time.time()
elapsed_time = end_time - start_time
print(answer)
print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds\n")
```
- **end_time**: Marca o final da execução
- **elapsed_time**: Calcula o tempo total gasto
- **Prints**: Exibe a resposta completa e o tempo de execução formatado

---

## 🔄 Fluxo Completo da Aplicação

```
1. Carregar variáveis de ambiente (chave API)
   ↓
2. Inicializar modelo ChatOpenAI
   ↓
3. Receber interesse do usuário ("beaches")
   ↓
4. Chain 1 - Recomendar Cidade
   - Executa: promptCity → model → cityParser
   - Retorna: Cidade e motivo em JSON
   ↓
5. Chain 2 - Recomendar Restaurantes
   - Executa: promptRestaurant → model → restaurantParser
   - Usa a cidade da Chain 1
   - Retorna: Restaurantes em JSON
   ↓
6. Chain 3 - Sugerir Atividades Culturais
   - Executa: promptCultural → model → StrOutputParser
   - Usa a cidade da Chain 1
   - Retorna: Atividades como texto livre
   ↓
7. Exibir resultado completo
   ↓
8. Mostrar tempo total de execução
```

---

## 🎯 Resumo da Arquitetura

| Componente | Função |
|---|---|
| **Imports** | Carregar dependências necessárias |
| **Classes Pydantic** | Validar e estruturar dados |
| **Parsers** | Converter saídas em formatos esperados |
| **Prompts** | Instruções dinâmicas para o modelo |
| **Modelo ChatOpenAI** | Motor de inteligência artificial |
| **Chains** | Pipelines que orquestram prompt → modelo → parser |
| **Main Chain** | Execução sequencial de todas as chains |
| **Resultado** | Resposta integrada com cronometragem |

---

## 💡 Conceitos-Chave

- **LangChain**: Framework que simplifica construção de aplicações LLM
- **Prompt Template**: Prompts reutilizáveis com variáveis dinâmicas
- **Parsers**: Garantem que as respostas estejam no formato esperado
- **Pydantic**: Valida tipos de dados em tempo de execução
- **Chains**: Pipelines que encadeiam transformações sequencialmente
- **Temperature**: Parâmetro que controla criatividade do modelo
