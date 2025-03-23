import os
import requests
import fitz  # PyMuPDF
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings as SentenceTransformerEmbeddings


from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if mistral_api_key:
    print(f"✅ Mistral API Key Loaded: {mistral_api_key[:5]}********")
else:
    print("❌ API Key NOT Found!")
    exit()

# ========== 1. Search ArXiv ==========
def search_arxiv(query, max_results=10):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Error fetching ArXiv data: {response.status_code}")
        return []

    root = ET.fromstring(response.text)
    papers = []

    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        published_date = entry.find("{http://www.w3.org/2005/Atom}published").text[:10]

        year = int(published_date[:4])
        if year < 2015:
            continue

        pdf_url = None
        for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]

        arxiv_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]

        papers.append({
            "title": title,
            "summary": summary,
            "published_date": published_date,
            "pdf_url": pdf_url,
            "arxiv_id": arxiv_id,
        })

    return papers

# ========== 2. Get Citations from OpenAlex ==========
def get_citation_count(title):
    try:
        url = f"https://api.openalex.org/works?search={requests.utils.quote(title)}"
        res = requests.get(url).json()
        return res["results"][0].get("cited_by_count", 0) if res["results"] else 0
    except:
        return 0

# ========== 3. Download PDF & Extract Text ==========
def pdf_to_text(pdf_url):
    try:
        response = requests.get(pdf_url)
        with open("paper.pdf", "wb") as f:
            f.write(response.content)
        print("📥 PDF Downloaded: paper.pdf")

        doc = fitz.open("paper.pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        return f"❌ PDF Error: {e}"

# ========== 4. Chunking ==========
def chunk_text(text, chunk_size=512, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# ========== 5. Embedding ==========
def embed_chunks(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=embeddings)

# ========== 6. RAG Q&A with Mistral via OpenAI-compatible API ==========
def start_qa_chain(vectorstore):
    llm = ChatOpenAI(
        model="mistral-small",  # Or "mistral-medium", "mistral-large"
        openai_api_key=mistral_api_key,
        openai_api_base="https://api.mistral.ai/v1",  # ✅ This is key
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
    )

    print("\n🤖 Ready for Q&A. Type 'exit' to quit.")
    while True:
        question = input("💬 Ask a question: ")
        if question.lower() == "exit":
            break
        answer = qa.run(question)
        print(f"📖 Answer: {answer}\n")

# ========== MAIN ==========
if __name__ == "__main__":
    query = input("🔍 Enter your paper query: ")
    papers = search_arxiv(query, max_results=15)

    if not papers:
        print("❌ No papers found.")
        exit()

    print(f"✅ Found {len(papers)} papers. Fetching citations...\n📚 Papers found:")
    for i, paper in enumerate(papers):
        paper["citations"] = get_citation_count(paper["title"])
    papers.sort(key=lambda x: x["citations"], reverse=True)

    for i, paper in enumerate(papers):
        print(f"{i + 1}. {paper['title']} ({paper['citations']} citations)")

    selected_title = input("\n✏️ Enter the exact title of the paper you want to chat with: ").strip()
    selected_paper = next((p for p in papers if p["title"].lower() == selected_title.lower()), None)

    if not selected_paper:
        print("❌ Paper not found.")
        exit()

    print(f"\n🔗 Processing: {selected_paper['title']}")
    pdf_text = pdf_to_text(selected_paper["pdf_url"])

    print("📚 Splitting and embedding chunks...")
    chunks = chunk_text(pdf_text)
    vectorstore = embed_chunks(chunks)

    start_qa_chain(vectorstore)
