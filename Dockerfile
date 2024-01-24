FROM python:3.10.10

EXPOSE 8501

# Configure o diretório de trabalho
WORKDIR /app

# Copie o restante do código para o diretório de trabalho
COPY . .

# Instale o Streamlit
RUN pip install -r requirements.txt

#ENTRYPOINT [ "streamlit", "run"]

#CMD streamlit run app/frontend.py
# Comando para executar o aplicativo Streamlit
CMD ["streamlit", "run","--server.address","0.0.0.0","--server.port","8501", "app/load_recursive.py"]

#CMD [ "app/frontend.py" ]