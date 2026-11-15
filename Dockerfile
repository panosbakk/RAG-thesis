FROM python:3.12


WORKDIR /rag

#Εγκατάσταση εξαρτήσεων
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Αντιγραφή του κώδικα (συμπεριλαμβανομένου του rag.py) στον τρέχοντα φάκελο /rag
COPY . .

#Άνοιγμα θύρας για το Streamlit UI
EXPOSE 8501

#Εντολή τρεξίματος της εφαρμογής
CMD ["streamlit", "run", "rag.py", "--server.address=0.0.0.0", "--server.port=8501"]FROM python:3.10-slim

