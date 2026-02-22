# Documentação: basic_chat_v2.py

## 📋 Visão Geral

O arquivo `basic_chat_v2.py` implementa um **chatbot conversacional com memória** que atua como um guia de viagem especializado em destinos brasileiros. Utiliza LangChain para orquestrar a integração entre modelos de IA (OpenAI) e gerenciamento de histórico de conversas.

## 🔄 Lógica do Código Passo a Passo

### 1. **Configuração Inicial**
```python
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```
- Carrega a chave de API do arquivo `.env`
- Armazena na variável `api_key` para uso posterior

### 2. **Inicialização do Modelo**
```python
modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=SecretStr(api_key) if api_key else None
)
```
- Cria uma instância do modelo ChatGPT 3.5 Turbo
- Temperatura 0.5 balanceia criatividade e consistência
- A chave de API é encapsulada por segurança

### 3. **Definição do Prompt Template**
```python
prompt_sugestao = ChatPromptTemplate.from_messages([
    ("system", "Você é um guia de viagem..."),
    ("placeholder", "{historico}"),
    ("human", "{query}")
])
```
- Estrutura a conversa em 3 partes:
  - **System**: Define personalidade da IA
  - **Placeholder**: Reserva espaço para histórico
  - **Human**: Pergunta do usuário

### 4. **Construção da Cadeia**
```python
cadeia = prompt_sugestao | modelo | StrOutputParser()
```
- Usa o operador `|` (pipe) do LangChain para conectar componentes
- **Fluxo**: Template → Modelo → Parser
- A saída de um componente vira entrada do próximo

### 5. **Gerenciamento de Memória**
```python
memoria = {}
sessao = "aula_langchain_alura"

def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]
```
- Dicionário `memoria` armazena históricos por sessão
- Função `historico_por_sessao()` retorna o histórico da sessão ou cria um novo
- Permite múltiplas conversas independentes

### 6. **Adição de Histórico à Cadeia**
```python
cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)
```
- Encapsula a cadeia anterior com capacidade de memória
- `input_messages_key`: Especifica qual dado é a entrada do usuário
- `history_messages_key`: Especifica onde inserir o histórico

### 7. **Execução das Perguntas**
```python
for uma_pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke({
        "query": uma_pergunta
    }, config={"configurable": {"session_id": sessao}})
    print("Usuário: ", uma_pergunta)
    print("IA: ", resposta, "\n")
```
- Loop executa cada pergunta sequencialmente
- `.invoke()`: Executa a cadeia com a pergunta
- `session_id`: Identifica a sessão para associar ao histórico correto
- Printa pergunta e resposta formatadas

---

## 💡 Como Usar Este Código

### Pré-requisitos
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Crie um arquivo `.env` na raiz do projeto:
   ```
   OPENAI_API_KEY=sua_chave_aqui
   ```

### Executar
```bash
python basic_chat_v2.py
```

### Saída Esperada
```
Usuário: Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?
IA: Olá! Aqui é o Sr. Passeios... [resposta do modelo]

Usuário: Qual a melhor época do ano para ir?
IA: A melhor época... [resposta com contexto da pergunta anterior]
```

---

## 🎯 Fluxo de Execução Visual

```
┌─────────────────────────────────────────┐
│ 1. Carrega Variáveis de Ambiente (.env) │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 2. Inicializa ChatOpenAI (gpt-3.5-turbo)│
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 3. Define ChatPromptTemplate            │
│    (System + Histórico + Query)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 4. Cria Cadeia:                         │
│    Template → Modelo → Parser           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 5. Envolve Cadeia com Gerenciador de    │
│    Histórico (RunnableWithMessageHistory)│
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 6. Para Cada Pergunta:                  │
│    - Adiciona ao Histórico              │
│    - Processa com Contexto              │
│    - Exibe Resposta                     │
└─────────────────────────────────────────┘
```

---

## 🔑 Conceitos-Chave

| Conceito | Explicação |
|----------|-----------|
| **Cadeia (Chain)** | Sequência de operações conectadas onde a saída de uma é entrada da próxima |
| **Prompt Template** | Estrutura reutilizável para criar mensagens com placeholders |
| **Histórico de Mensagens** | Registro de conversas anteriores usado como contexto |
| **Sessão** | Identificador único para separar conversas independentes |
| **Temperature** | Parâmetro que controla aleatoriedade (0 = determinístico, 1 = aleatório) |

---

## 📊 Melhorias Possíveis

1. **Persistência**: Substituir `InMemoryChatMessageHistory` por banco de dados para manter histórico permanente
2. **Modelos Alternativos**: Trocar `gpt-3.5-turbo` por `gpt-4` para respostas mais sofisticadas
3. **Tratamento de Erros**: Adicionar try/except para falhas de API
4. **Validação de Input**: Sanitizar e validar entradas do usuário
5. **Logging**: Registrar conversas para auditoria e análise

---

## 📚 Documentação Oficial

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [Pydantic Documentation](https://docs.pydantic.dev/)
