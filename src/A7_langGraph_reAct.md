# Documentação do Arquivo: A7_langGraph_reAct

## Visão Geral

Este arquivo implementa um fluxo de trabalho baseado em um grafo de estados utilizando a biblioteca `langgraph` e ferramentas da `langchain`. Ele demonstra como criar um agente de IA que utiliza ferramentas para realizar cálculos matemáticos simples (adição, subtração e multiplicação) e gerencia o consumo de tokens durante o processamento de mensagens.

## Estrutura do Código

### 1. **Dependências e Configuração**
O código importa diversas bibliotecas, incluindo:
- `langchain_core` para manipulação de mensagens e ferramentas.
- `langgraph` para criar e gerenciar o grafo de estados.
- `dotenv` para carregar variáveis de ambiente, como a chave da API da OpenAI.

A chave da API da OpenAI é carregada a partir de um arquivo `.env`:
```python
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

### 2. **Definição do Estado do Agente**
O estado do agente é definido como um dicionário tipado (`TypedDict`) contendo:
- `messages`: Uma sequência de mensagens baseadas no modelo `BaseMessage`.
- `tokens_used` e `total_tokens`: Campos opcionais para rastrear o consumo de tokens.

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tokens_used: NotRequired[int]
    total_tokens: NotRequired[int]
```

### 3. **Ferramentas**
Três ferramentas são definidas para realizar operações matemáticas básicas:
- `add_tool`: Soma dois números.
- `subtractor_tool`: Subtrai dois números.
- `multiply_tool`: Multiplica dois números.

Cada ferramenta é decorada com `@tool`, o que as torna utilizáveis pelo agente.

```python
@tool
def add_tool(a: int, b: int) -> int:
    return a + b
```

As ferramentas são agrupadas em uma lista:
```python
tools = [add_tool, subtractor_tool, multiply_tool]
```

### 4. **Modelo de Linguagem**
O modelo de linguagem utilizado é o `ChatOpenAI`, configurado com o modelo `gpt-4.1-nano` e uma temperatura de 0.7. As ferramentas são vinculadas ao modelo:
```python
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7,
    api_key=SecretStr(api_key) if api_key else None,
).bind_tools(tools)
```

### 5. **Nós do Grafo**
#### **Nó de Chamada do Modelo**
O nó `model_call` processa mensagens do estado do agente, adiciona um prompt do sistema e invoca o modelo de linguagem. Ele também rastreia o consumo de tokens:
```python
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    messages = [system_prompt, *state["messages"]]
    response = llm.invoke(messages)
    state["tokens_used"] = response.response_metadata["token_usage"]["total_tokens"]
    state["total_tokens"] = state.get("total_tokens", 0) + state["tokens_used"]
    state["messages"] = [response]
    return state
```

#### **Nó de Ferramentas**
O nó `tool_node` utiliza as ferramentas definidas anteriormente para realizar operações matemáticas.

### 6. **Condições de Continuação**
A função `should_continue` verifica se o último tipo de mensagem suporta chamadas de ferramentas. Se sim, o fluxo continua para o nó de ferramentas; caso contrário, o grafo termina:
```python
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue_edge"
    return "end_edge"
```

### 7. **Definição do Grafo**
O grafo é definido utilizando a classe `StateGraph`. Ele conecta os nós e define as condições de transição:
```python
graph = StateGraph(AgentState)
graph.add_node("model_call_node", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "model_call_node")
graph.add_conditional_edges(
    "model_call_node", should_continue, {"continue_edge": "tool_node", "end_edge": END}
)
graph.add_edge("tool_node", "model_call_node")
app = graph.compile()
```

### 8. **Execução**
O fluxo é iniciado com um estado inicial contendo uma mensagem do usuário. O método `app.stream` executa o grafo e imprime as mensagens processadas:
```python
inputs: AgentState = {
    "messages": [
        HumanMessage(content="Add 40 + 12 and then multiply the result by 3.")
    ],
}
print_stream(app.stream(inputs, stream_mode="values"))
```

## Consumo de Tokens
O consumo de tokens é rastreado em dois níveis:
1. **Tokens usados no nó atual**: Armazenados em `state["tokens_used"]`.
2. **Tokens totais**: Acumulados em `state["total_tokens"]`.

Essas informações são úteis para monitorar custos e otimizar o uso do modelo.

## Resumo do Fluxo
1. O usuário envia uma mensagem inicial.
2. O nó `model_call` processa a mensagem e invoca o modelo de linguagem.
3. Se o modelo indicar a necessidade de ferramentas, o fluxo continua para o nó `tool_node`.
4. O nó `tool_node` executa as ferramentas necessárias e retorna ao nó `model_call`.
5. O fluxo termina quando nenhuma ferramenta adicional é necessária.

## Conclusão
Este arquivo demonstra como integrar um modelo de linguagem com ferramentas personalizadas em um fluxo baseado em grafo. Ele destaca o uso eficiente de tokens e a flexibilidade do `langgraph` para criar fluxos dinâmicos e interativos.