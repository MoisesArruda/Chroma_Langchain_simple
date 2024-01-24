''' Este arquivo foi criado com o objetivo de extrair as informações de um documento PDF e passa-las para um TEXTO /
para que o modelo possa armazenar os chunks e transmitir a resposta para o usuário e para um arquivo .JSON.
Este projeto utiliza as seguintes funções e métodos:
    
    PdfReader ==> Carrega o documento, indicado quando se deseja trabalhar com textos.
    
    extract_text ==> Extrair o texto do documento.
    
    RecursiveCharacterTextSplitter ==> Indicada para arquivos com assuntos mais complexos, possui maior ajustabilidade.
        Para divisão básica em fragmentos e acesso direto ao texto com possibilidade de sobreposição /
        de fragmentos(chunk_overlap) para melhor aprendizagem.
    
    split_text ==> Separa o texto em chunks.
    
    Chroma.from_texts ==> Armazena os chunks dentro do Vector DB.

    ConversationBufferWindowMemory ==> Armazena apenas as interações definidas pelo K, ideal quando o histórico/
        completo não é necessário, ou a memória é limitada'''

import timeit

from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory,FileChatMessageHistory
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter,TextSplitter
from class_func import create_chat,create_embeddings,define_pastas,verify_pdf,create_prompts
from PyPDF2 import PdfReader
from langchain.chains import LLMChain
from langchain.cache import InMemoryCache
from langchain.chains import RetrievalQA

llm_chat = create_chat()
llm_embeddings = create_embeddings()
prompt = create_prompts()
file = define_pastas(r'data')
print(file)

try:
    # O [0] é para não retornar como lista, apenas o nome
    data = verify_pdf(file)[0]
    print(data)
except (FileNotFoundError, ValueError) as e:
    print(e)

# Passar o caminho completo do arquivo, fazer a leitura e retornar como lista
pdf_loader = PdfReader(f'{file}/{data}')

# Armazenar todo o texto do documento em uma variável para realizar o processamento
text = ""
for page in pdf_loader.pages:
    text += page.extract_text()

# Criação do objeto RecursiveCharacterTextSplitter para gerar o chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Tamanho do chunk/texto que vai estar ali dentro
    chunk_size=1000,
    # Tamanho da sobreposição, conseguindo aproveitar palavras/frases que foram divididas
    chunk_overlap=200,
    # Medir o comprimento
    length_function=len
)

# Realizar a divisão do Texto e gerar os chunks
chunks = text_splitter.split_text(text)

# Realizar o armazenamento do tempo de resposta
llm_cache = InMemoryCache()

# Criação do objeto Chroma, para armazenar os chunks
db = Chroma.from_texts(
    texts=chunks,
    embedding=llm_embeddings,
    cache=llm_cache
)

query = "Como utilizar o docker?"

# Realizar a busca por similirdade, retorna a quantidade especificada pelo K
contexto = db.similarity_search(query, k=3)

# Criação do objeto ConversationBufferWindowMemory para armazenar o chat
memory = ConversationBufferWindowMemory(
    # Armazena o histórico da interação com o modelo
    memory_key="chat_history",
    # Chave do input
    input_key="query",
    # Quantidade de interações que serão "armazenadas" para continuar o contexto da conversa.
    k=2,
    #
    return_messages=True,
    # Armazena as respostas em um arquivo .json
    chat_memory=FileChatMessageHistory(file_path="messages_recursive.json"),
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
# Método simples que responde a pergunta utilizando o modelo, maneira mais rápida e fácil.
response = llm_chain.run(query=query,contexto=contexto,memory=memory)
print(response) 

# Cria um objeto RetrievalQA, utilizará o LLM para entender a pergunta
# Classe que permite personalizar o processo de Pergunta e resposta, dando mais controle sobre a qualidade e a precisão das respostas.
qa_chain = RetrievalQA.from_chain_type(
    # Modelo que será utilizado
    llm=llm_chat,
    # Tipo de cadeia para responder perguntas factuais
    chain_type="stuff",
    # Deve recuperar apenas os dois resultados principais
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    # Gerar os logs de resposta
    verbose=True,
)

# Finaliza a contagem do tempo de resposta
elapsed_time = timeit.default_timer() - start_time

print(qa_chain(query))
print(f"Executado em {elapsed_time} segundos")

