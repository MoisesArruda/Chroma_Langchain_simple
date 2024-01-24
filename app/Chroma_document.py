'''Este arquivo foi criado com o objetivo de extrair as informações de um documento PDF, para que o modelo possa /
armazenar os chunks, realizar o aprendizado, e transmitir a resposta para o usuário e para um arquivo .JSON.
Este projeto utiliza as seguintes funções e métodos:

    PyPDFLoader ==> Carrega o documento, indicado quando se pretende trabalhar com PDF.

    load ==> Realiza o carregamento do arquivo e permite o aproveitamento de metadados. Retorna uma lista de objetos /
        Document, onde cada representa uma página do PDF. Melhor para situações onde o arquivo trata de diversos assuntos /
        ou arquivos menores

    load_and_split ==> Realiza o carregamento do arquivo, retornando uma lista de listas, onde cada lista representa um /
        assunto do PDF. Melhor para situações onde o arquivo é maior, pois só armazena o texto dos assuntos relevantes.

    CharacterTextSplitter ==> Indicada para arquivos menores ou com menor complexidade.
        Para divisão básica em fragmentos e acesso direto ao texto /
        com possibilidade de sobreposição de fragmentos(chunk_overlap) para melhor aprendizagem

    split_documents ==> Separa o conteúdo do PDF em chunks.

    Chroma.from_documents ==> Armazenar os chunks dentro do Vector DB.
    
    ConversationBufferMemory = Armazena o histórico completo da conversa em uma lista, ideal para chatbots/
        onde é necessário lembrar o que foi dito anteriormente para responder as instruções.'''


import timeit

from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory,FileChatMessageHistory
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from class_func import create_chat,create_embeddings,define_pastas,verify_pdf,create_prompts
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.cache import InMemoryCache
from langchain.chains import RetrievalQA


llm_chat = create_chat()
llm_embeddings = create_embeddings()
prompt = create_prompts()
file = define_pastas(data=r'C:\AI\DevIA\Chroma\data')

try:
    # O [0] é para não retornar como lista, apenas o nome
    data = verify_pdf(file)[0]
    print(data)
except (FileNotFoundError, ValueError) as e:
    print(e)


# Cria um objeto passando o caminho completo do arquivo.
pdf_loader = PyPDFLoader(f'{file}/{data}')

# Método load carregar o PDF, é possível realizar o aproveitamento de metadados
pages = pdf_loader.load()

# Criação do objeto CharacterTextSplitter para gerar o chunks
text_splitter = CharacterTextSplitter(
    # Tamanho do chunk/texto que vai estar ali dentro
    chunk_size=1000,
    # Tamanho da sobreposição, conseguindo aproveitar palavras/frases que foram divididas
    chunk_overlap=200,
    # Medir o comprimento
    length_function=len
)

# Criação do chunks a partir da divisão de páginas
chunks = text_splitter.split_documents(pages)

# Realizar o armazenamento do tempo de resposta
llm_cache = InMemoryCache()

# Realizar o armazenamento no DB
db = Chroma.from_documents(documents=chunks, embedding=llm_embeddings,cache=llm_cache)

query = "O que é o Docker?"

# Contexto da resposta, pesquisa por similaridade
contexto = db.similarity_search(query, k=3)

# Criação do objeto ConversationBufferMemory para armazenar o chat
memory = ConversationBufferMemory(
    # Armazena as respostas em um arquivo .json
    chat_memory=FileChatMessageHistory(file_path="messages_load.json"),
    # Armazena o histórico da interação com o modelo
    memory_key="chat_history",
    # Chave do input
    input_key="query",
    # Tras a pesquisa por similaridade do conteúdo que gerou a resposta
    contexto=contexto
)

# Criação do objeto LLMChain, cadeia de conexão para gerar a resposta do LLM
llm_chain=LLMChain(
    # Modelo que será utilizado
    llm=llm_chat,
    # Definição do prompt utilizado
    prompt=prompt,
    # Retornar logs
    verbose=True,
    # Configuração da memória que será utilizada
    memory=memory
)

# Iniciar contagem do tempo de resposta
start_time = timeit.default_timer()

# Gerar a resposta do modelo, vai me retornar um objeto chain. Posteriormente será aproveitado na qa_chain
response = llm_chain.run(query=query,contexto=contexto,memory=memory)
print(response)

# Cria um objeto RetrievalQA, utilizará o LLM para entender a pergunta
qa_chain = RetrievalQA.from_chain_type(
    # Modelo que será utilizado
    llm=llm_chat,
    # Tipo de cadeia para responder perguntas factuais
    chain_type="stuff",
    # Deve recuperar apenas os dois resultados principais
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    # Gerar os logs de resposta
    verbose=True
)

# Finaliza a contagem do tempo de resposta
elapsed_time = timeit.default_timer() - start_time

print(qa_chain(query))
print(f"Executado em {elapsed_time} segundos")

