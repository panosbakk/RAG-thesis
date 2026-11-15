FROM python:3.12

# Ορίζεις τον φάκελο εργασίας μέσα στο container (μπορείς να τον πεις /rag)
WORKDIR /rag

# Εγκατάσταση εξαρτήσεων
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγραφή του κώδικα (συμπεριλαμβανομένου του rag.py) στον τρέχοντα φάκελο /rag
COPY . .

# Άνοιγμα θύρας για το Streamlit UI
EXPOSE 8501

# Προσοχή εδώ: Δηλώνεις ρητά να τρέξει το rag.py αντί για app.py
CMD ["streamlit", "run", "rag.py", "--server.address=0.0.0.0", "--server.port=8501"]FROM python:3.10-slim

# Ορίζεις τον φάκελο εργασίας μέσα στο container (μπορείς να τον πεις /rag)
WORKDIR /rag

# Εγκατάσταση εξαρτήσεων
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγραφή του κώδικα (συμπεριλαμβανομένου του rag.py) στον τρέχοντα φάκελο /rag
COPY . .

# Άνοιγμα θύρας για το Streamlit UI
EXPOSE 8501

# Προσοχή εδώ: Δηλώνεις ρητά να τρέξει το rag.py αντί για app.py
CMD ["streamlit", "run", "rag.py", "--server.address=0.0.0.0", "--server.port=8501"]

# Ορίζεις τον φάκελο εργασίας μέσα στο container (μπορείς να τον πεις /rag)
WORKDIR /rag

# Εγκατάσταση εξαρτήσεων
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγραφή του κώδικα (συμπεριλαμβανομένου του rag.py) στον τρέχοντα φάκελο /rag
COPY . .

# Άνοιγμα θύρας για το Streamlit UI
EXPOSE 8501

# Προσοχή εδώ: Δηλώνεις ρητά να τρέξει το rag.py αντί για app.py
CMD ["streamlit", "run", "rag.py", "--server.address=0.0.0.0", "--server.port=8501"]