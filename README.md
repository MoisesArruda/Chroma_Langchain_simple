# Explicação do projeto.

Este repositório tem o propósito de armazenar parte do conhecimento adquirido durante os estudos e desenvolvimento
do projeto utilizando LangChain e o Vector DataBase Chroma.
O grande objetivo é facilitar futuras consultas através de scripts.py simples e intuitivos, 

## Explicação Chroma_Text.py

[Acesse o script neste link](https://github.com/MoisesArruda/Chroma_Langchain_simple/blob/master/app/Chroma_Text.py)

Este arquivo foi criado com o objetivo de extrair as informações de um documento PDF e passa-las para um *TEXTO*
para que o modelo possa armazenar os chunks e transmitir a resposta para o usuário e para um arquivo *.JSON*.

**Este projeto utiliza as seguintes funções e métodos:**
* **PdfReader** ==> Carrega o documento, indicado quando se deseja trabalhar com textos.
* **extract_text** ==> Extrair o texto do documento.
* **RecursiveCharacterTextSplitter** ==> Indicada para arquivos com assuntos mais complexos, possui maior ajustabilidade.
    Para divisão básica em fragmentos e acesso direto ao texto com possibilidade de sobreposição 
    de fragmentos(chunk_overlap) para melhor aprendizagem.
* **split_text** ==> Separa o texto em chunks.
* **Chroma.from_texts** ==> Armazenar os chunks dentro do Vector DB.
* **ConversationBufferWindowMemory** ==> Armazena apenas as interações definidas pelo K, ideal quando o histórico 
    completo não é necessário, ou a memória é limitada.
* **run** = Execute o modelo, vai me retornar um objeto chain. Posteriormente será aproveitado na qa_chain.
* **RetrievalQA** ==> Responde a pergunta utilizando o modelo, maneira mais personalizavel.

## Explicação Chroma_document.py

[Acesse o script neste link](https://github.com/MoisesArruda/Chroma_Langchain_simple/blob/master/app/Chroma_document.py)

Este arquivo foi criado com o objetivo de extrair as informações de um documento PDF, para que o modelo possa
armazenar os chunks, realizar o aprendizado, e transmitir a resposta para o usuário e para um arquivo .JSON.

**Este projeto utiliza as seguintes funções e métodos:**
* **PyPDFLoader** ==> Carrega o documento, indicado quando se pretende trabalhar com PDF.
* **load** ==> Realiza o carregamento do arquivo e permite o aproveitamento de metadados. Retorna uma lista de objetos
    *Document*, onde cada representa uma página do PDF. Melhor para situações onde o arquivo trata de diversos assuntos
    ou arquivos menores.
* **load_and_split** ==> Realiza o carregamento do arquivo, retornando uma lista de listas, onde cada lista representa um
    assunto do PDF. Melhor para situações onde o arquivo é maior, pois só armazena o texto dos assuntos relevantes.
* **CharacterTextSplitter** ==> Indicada para arquivos menores ou com menor complexidade.
    Para divisão básica em fragmentos e acesso direto ao texto com possibilidade de sobreposição de fragmentos(chunk_overlap) 
    para melhor aprendizagem.
* **split_documents** ==> Separa o conteúdo do PDF em chunks.
* **Chroma.from_documents** ==> Armazenar os chunks dentro do Vector DB.
* **ConversationBufferMemory** = Armazena o histórico completo da conversa em uma lista, ideal para chatbots onde é necessário lembrar
    o que foi dito anteriormente para responder as instruções.
* **run** = Execute o modelo, vai me retornar um objeto chain. Posteriormente será aproveitado na qa_chain.
* **RetrievalQA** ==> Responde a pergunta utilizando o modelo, maneira mais personalizavel.