# Naive RAG Chatbot - Phase 1 Bootcamp Project

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, HuggingFace transformers, and Chroma vector database. This project demonstrates a complete RAG pipeline for document-based question answering.

## 🎯 Project Overview

This RAG chatbot can answer questions based on PDF documents by:
1. Loading and chunking PDF documents
2. Creating vector embeddings using sentence-transformers
3. Storing embeddings in a Chroma vector database
4. Retrieving relevant context for user queries
5. Generating answers using a local language model

## 🏗️ Architecture

- **Document Loader**: PyPDFLoader for PDF processing
- **Text Splitter**: RecursiveCharacterTextSplitter for chunking
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: Chroma DB for similarity search
- **LLM**: DistilGPT2 for text generation
- **Framework**: LangChain for orchestration

## 📋 Requirements

### System Requirements
- Python 3.8+
- At least 4GB RAM (for models)
- Internet connection (for initial model downloads)

### Dependencies
```bash
langchain
langchain-community
langchain-huggingface
pypdf
sentence-transformers
chromadb
transformers
torch
ipywidgets
python-dotenv
```

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd naive-rag-chatbot
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create environment file**
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 📁 Project Structure

```
naive-rag-chatbot/
├── Naive_RAG_Notebook.ipynb    # Main notebook
├── uploaded_files/             # PDF documents directory
│   └── kenya-market-update.pdf # Sample document
├── chroma_store/              # Vector database (created after first run)
├── .env                       # Environment variables
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🖥️ Usage

### Option 1: Jupyter Notebook (Recommended)
1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `Naive_RAG_Notebook.ipynb`

3. Run all cells sequentially

4. Use the interactive widget at the bottom to ask questions

### Option 2: Console Interface
Run the notebook cells up to the console loop section and interact via command line.

## 📄 Adding Your Documents

1. Place your PDF files in the `uploaded_files/` directory
2. Update the `pdf_path` variable in the notebook:
```python
pdf_path = "uploaded_files/your-document.pdf"
```
3. Re-run the document ingestion cells

## 🔧 Configuration

### Chunk Size and Overlap
Adjust text splitting parameters:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Characters per chunk
    chunk_overlap=200   # Overlap between chunks
)
```

### Retrieval Settings
Modify number of retrieved documents:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Top 3 results
```

### Model Parameters
Adjust text generation settings:
```python
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,  # Maximum response length
    device=-1           # CPU usage
)
```

## 🎮 Example Queries

Try asking questions like:
- "What are the key market trends mentioned in the document?"
- "What information is provided about real estate performance?"
- "Summarize the main findings of the report"

## 🔍 How It Works

1. **Document Processing**: PDFs are loaded and split into manageable chunks
2. **Embedding Creation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in Chroma database for fast retrieval
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Context Retrieval**: Most relevant chunks are retrieved based on similarity
6. **Answer Generation**: LLM generates responses using retrieved context

## ⚡ Performance Notes

- **First Run**: Models will be downloaded and cached (may take time)
- **Subsequent Runs**: Models load from local cache (faster)
- **CPU Usage**: Configured to use CPU by default (slower but more compatible)
- **Memory**: Embeddings and models require significant RAM

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Timeout**
   - Solution: Run the pre-download cells separately
   - Models are cached in `~/.cache/huggingface/hub/`

2. **Memory Issues**
   - Reduce `chunk_size` and `max_new_tokens`
   - Close other applications to free RAM

3. **No Answer Generated**
   - Check if PDF was loaded correctly
   - Verify vector store has content
   - Try rephrasing your question

4. **Import Errors**
   - Ensure all dependencies are installed
   - Update langchain packages: `pip install -U langchain-huggingface`

## 🔮 Future Improvements

- [ ] Support for multiple document formats (DOCX, TXT)
- [ ] Better chunking strategies
- [ ] Advanced retrieval techniques (hybrid search)
- [ ] Conversation memory
- [ ] Web interface with Streamlit/Gradio
- [ ] GPU optimization
- [ ] Evaluation metrics

## 📖 Learning Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [Chroma DB Guide](https://docs.trychroma.com/)
- [RAG Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Acknowledgments

- NSK AI RAG Bootcamp 2025
- LangChain community
- HuggingFace team for open-source models
- Chroma team for the vector database
