import os
import pandas as pd
import re
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

#Configuration
os.environ["OPENAI_API_KEY"] = '<API_KEY>'
LOAN_APPLICATIONS_FILE = 'loan_applications.csv'
CREDIT_POLICY_DIR = 'credit_policy_documents'
VECTOR_DB_PATH = 'faiss_policy_index'
LLM_MODEL = "gpt-4o-mini"

#RAG set up
def setup_policy_knowledge_base():
    documents = []
    if not os.path.exists(CREDIT_POLICY_DIR):
        raise FileNotFoundError(f"Policy directory '{CREDIT_POLICY_DIR}' not found.")

    for filename in os.listdir(CREDIT_POLICY_DIR):
        if filename.endswith('.txt'):
            loader = TextLoader(os.path.join(CREDIT_POLICY_DIR, filename), encoding='utf-8')
            documents.extend(loader.load())
    
    if not documents:
        raise ValueError("No .txt documents found in policy directory.")

    texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
    vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())
    vectorstore.save_local(VECTOR_DB_PATH)
    return vectorstore

def check_compliance(vectorstore, loan_details):
    query_text = (f"Loan Details: Income=${loan_details.get('income', 0):,}, "
                  f"DTI={loan_details.get('debt_to_income_ratio', 0.0):.1f}%, "
                  f"Credit Score={loan_details.get('credit_score', 0)}, "
                  f"Purpose='{loan_details.get('purpose_of_loan_simplified', 'N/A')}'. "
                  "Is this compliant? Explain and cite.")

    compliance_prompt = ChatPromptTemplate.from_messages([
        ("system", "Assess the loan against the policy context. Output format: 'Status: [Compliant/Non-Compliant/Requires Review] - [Reason]\nReference: [Source]'\n\nContext:\n{context}"),
        ("human", "{input}")
    ])
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)
    document_chain = create_stuff_documents_chain(llm, compliance_prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    try:
        response = retrieval_chain.invoke({"input": query_text})
        raw_output = response.get('answer', '')
        status = re.search(r"Status:\s*(.*)", raw_output)
        ref = re.search(r"Reference:\s*(.*)", raw_output)
        return (status.group(1).strip() if status else "Parse Error", 
                ref.group(1).strip() if ref else "N/A")
    except Exception as e:
        return f"API Error: {e}", "N/A"

# Main Execution
if __name__ == "__main__":
    if os.environ["OPENAI_API_KEY"] == '<REPLACE_WITH_YOUR_API_KEY>':
        print("ERROR: OpenAI API Key is not set.")
        exit()

    try:
        policy_kb = setup_policy_knowledge_base()
        df_loans = pd.read_csv(LOAN_APPLICATIONS_FILE)
    except (FileNotFoundError, ValueError) as e:
        print(f"Setup Error: {e}")
        exit()

    for _, row in df_loans.iterrows():
        status, citation = check_compliance(policy_kb, row.to_dict())
        print(f"Loan ID: {row.get('loan_id', 'N/A')}\n  Status: {status}\n  Ref: {citation}")
