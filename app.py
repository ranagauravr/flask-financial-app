import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from io import BytesIO
import markdown
import base64
import matplotlib
matplotlib.use('Agg')  # ✅ Must be before importing pyplot
import matplotlib.pyplot as plt
import google.generativeai as genai
from PyPDF2 import PdfReader
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.decomposition import PCA
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

api_key = os.getenv("GEMINI_API_KEY")




app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = "your-secret-key"

# 🌐 Global in-memory store
uploaded_files_global = []
api_key_global = ""
llm_choice_global = "gemini-1.5-flash"

# 📄 PDF utilities
def get_pdf_page_count(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    return len(reader.pages)

def chunk_pages(total_pages, chunk_size=20):
    return [(start, min(start + chunk_size - 1, total_pages)) for start in range(1, total_pages + 1, chunk_size)]

def extract_mixed_content(file_bytes_data, page_range):
    try:
        start, end = page_range
        pages = f"{start}-{end}"
        return partition_pdf(
            file=BytesIO(file_bytes_data),
            strategy="fast",
            infer_table_structure=True,
            pages=pages,
            extract_images=False,
            include_page_breaks=False
        )
    except Exception as e:
        print(f"[extract_mixed_content] ERROR on pages {page_range}: {e}")
        return []

def chunk_texts(elements, chunk_size=10000, chunk_overlap=1000):
    texts = [str(e.text).strip() for e in elements if hasattr(e, 'text') and e.text and str(e.text).strip()]
    if not texts:
        return []
    full_text = "\n\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(full_text)

def build_faiss_index(chunks):
    docs = [
        Document(page_content=chunk, metadata={"source": f"chunk-{i}"})
        for i, chunk in enumerate(chunks)
    ]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_index = FAISS.from_documents(docs, embedding_model)
    return vector_index



# 📊 Render 2D chunk plot as base64 HTML
def render_chunk_length_plot(chunks):
    lengths = [len(c) for c in chunks if isinstance(c, str)]
    if not lengths:
        return ""

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(lengths, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("2D Chunk Length Distribution")
    ax.set_xlabel("Chunk Length (characters)")
    ax.set_ylabel("Frequency")

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)

    return f'<img src="data:image/png;base64,{img_base64}"/>'


def render_3d_embedding_plot(chunks):
    texts = [c for c in chunks if isinstance(c, str) and len(c.strip()) > 0]
    if len(texts) < 3:
        return "❗ Need at least 3 valid chunks for 3D visualization."

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = embeddings_model.embed_documents(texts)

    if len(vectors) < 3:
        return "❗ Not enough vectors for 3D PCA."

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(vectors)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], alpha=0.7, c='blue')
    ax.set_title("3D Embedding Visualization (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)

    return f'<img src="data:image/png;base64,{img_base64}"/>'






# 🧠 Main route
@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_files_global, api_key_global, llm_choice_global

    result = ""
    user_query = ""
    file_names = [f["filename"] for f in uploaded_files_global]

    if request.method == "POST":
        user_query = request.form.get("user_query", "")
        action = request.form.get("action")
        api_key = request.form.get("api_key") or api_key_global
        llm_choice = request.form.get("llm_choice") or llm_choice_global

        api_key_global = api_key
        llm_choice_global = llm_choice

        # Handle file upload
        files = request.files.getlist("pdf")[:3]
        if files and files[0].filename:
            uploaded_files_global = []
            for file in files:
                uploaded_files_global.append({
                    "filename": secure_filename(file.filename),
                    "content": file.read()
                })
            file_names = [f["filename"] for f in uploaded_files_global]

        if not uploaded_files_global:
            result = "❗ Please upload at least one PDF file."
            return render_template("index.html", files=[], result=result, user_query=user_query,
                                   api_key=api_key, llm_choice=llm_choice)

        if action == "get_response":
            all_elements = []
            all_chunks = []

            for f in uploaded_files_global:
                pdf_data = f["content"]
                total_pages = get_pdf_page_count(pdf_data)
                page_chunks = chunk_pages(total_pages)
                for chunk_range in page_chunks:
                    elements = extract_mixed_content(pdf_data, chunk_range)
                    all_elements.extend(elements)
                chunks = chunk_texts(all_elements)
                all_chunks.extend(chunks)

            if not all_chunks:
                result = "❗ No valid text chunks found to process."
            else:
                # Build FAISS index
                faiss_index = build_faiss_index(all_chunks)

                # Retrieve top 5 similar chunks
                retrieved_docs = faiss_index.similarity_search(user_query, k=5)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # Construct prompt with retrieved context
                prompt = f"""
You are a highly skilled and experienced financial analyst AI assistant.

You are given extracted content from financial reports. The content may include descriptions, figures, and also text representations of tables (where column alignment might not be perfect).
And do not hallucinate at all if you don't know the answer jsut deny politely and also do not answer any non financial questions. 


Context:
{context}

Now answer this question clearly and professionally:

\"{user_query}\"
"""

                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(llm_choice)
                    response = model.generate_content(prompt)
                    # result = f"💡 {response.text}"
                    markdown_html = markdown.markdown(response.text, extensions=['tables'])
                    result = f"<h4>💡 Gemini Answer:</h4>{markdown_html}"

                except Exception as e:
                    result = f"❗ Gemini LLM Error: {e}"

        elif action == "show_2d":
            all_elements = []
            all_chunks = []

            for f in uploaded_files_global:
                pdf_data = f["content"]
                total_pages = get_pdf_page_count(pdf_data)
                page_chunks = chunk_pages(total_pages)
                for chunk_range in page_chunks:
                    elements = extract_mixed_content(pdf_data, chunk_range)
                    all_elements.extend(elements)
                chunks = chunk_texts(all_elements)
                all_chunks.extend(chunks)

            if not all_chunks:
                result = "❗ No valid chunks found for visualization."
            else:
                result = render_chunk_length_plot(all_chunks)

        elif action == "show_3d":
            # result = "🌐 Simulated 3D embedding visualization."
            all_elements = []
            all_chunks = []

            for f in uploaded_files_global:
                pdf_data = f["content"]
                total_pages = get_pdf_page_count(pdf_data)
                page_chunks = chunk_pages(total_pages)
                for chunk_range in page_chunks:
                    elements = extract_mixed_content(pdf_data, chunk_range)
                    all_elements.extend(elements)
                chunks = chunk_texts(all_elements)
                all_chunks.extend(chunks)

            if not all_chunks:
                result = "❗ No valid chunks found for visualization."
            else:
                result = render_3d_embedding_plot(all_chunks)


    return render_template("index.html",
                           files=file_names,
                           result=result,
                           user_query=user_query,
                           api_key=api_key_global,
                           llm_choice=llm_choice_global)

if __name__ == "__main__":
    app.run(debug=True)
