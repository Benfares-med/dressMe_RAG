# 👗 DressMe AI Stylist

Assistant styliste IA basé sur RAG + Graph RAG + Agentic RAG.

## 🚀 Installation rapide

```bash
# 1. Cloner / créer le dossier
cd dressme

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer la clé API Groq
cp .env.example .env
# Edite .env et mets ta clé GROQ_API_KEY
# Obtenir une clé gratuite : https://console.groq.com

# 4. Lancer l'interface
streamlit run app/streamlit_app.py
```

## 🧪 Test en ligne de commande

```bash
python src/rag/rag_engine.py
```

## 📁 Structure

```
dressme/
├── data/
│   ├── documents/          ← Tes fichiers .txt de connaissances mode
│   └── chroma_db/          ← Vectorstore (auto-généré)
├── src/
│   ├── rag/
│   │   └── rag_engine.py   ← Moteur RAG principal ✅
│   ├── graph/              ← Graph RAG (Niveau 2) ⏳
│   └── agent/              ← Agentic RAG (Niveau 3) ⏳
├── app/
│   └── streamlit_app.py    ← Interface utilisateur ✅
├── .env.example
├── requirements.txt
└── README.md
```

## 🗺️ Roadmap

- [x] **Niveau 1** — RAG Classique (Groq + ChromaDB)
- [ ] **Niveau 2** — Graph RAG (Neo4j)
- [ ] **Niveau 3** — Agentic RAG (LangChain Agent)
