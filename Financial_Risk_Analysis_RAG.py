# Cell 3: Main RAG Compliance Checker Code

import os
import pandas as pd
from openai import OpenAI
import json
import re

# For RAG components
from langchain_community.document_loaders import TextLoader # For .txt files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS # Simple in-memory vector store
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

# --- Configuration ---
# API key is picked up from os.environ.get("OPENAI_API_KEY") which was set in Cell 2

HARDCODED_API_KEY = 'YOUR_OPENAI_API_KEY_HERE' # <--- PASTE YOUR KEY HERE
os.environ["OPENAI_API_KEY"] = HARDCODED_API_KEY

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define file paths (relative to the current Colab working directory)
# ENSURE these files are uploaded to your Colab session or mounted from Drive
LOAN_APPLICATIONS_FILE = 'loan_applications.csv'
CREDIT_POLICY_DIR = 'credit_policy_documents'
VECTOR_DB_PATH = 'faiss_policy_index' # Directory to save/load FAISS index (will be created)

LLM_MODEL = "gpt-4o-mini" # Or "gpt-4o" for higher quality/cost

# --- RAG Setup for Credit Policy Knowledge Base ---
def setup_rag_knowledge_base():
    """
    Loads credit policy documents, chunks them, creates embeddings, and builds a FAISS vector store.
    """
    # Check if a pre-built FAISS index exists
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(os.path.join(VECTOR_DB_PATH, 'index.faiss')):
        print(f"Loading existing FAISS index from {VECTOR_DB_PATH}...")
        embeddings = OpenAIEmbeddings() # Will pick up key from os.environ
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded.")
        return vectorstore

    print("Setting up RAG knowledge base from policy documents...")
    documents = []
    # Verify that the directory exists before trying to list its contents
    if not os.path.exists(CREDIT_POLICY_DIR):
        print(f"Error: Credit policy directory '{CREDIT_POLICY_DIR}' not found. Please upload it.")
        return None

    for filename in os.listdir(CREDIT_POLICY_DIR):
        filepath = os.path.join(CREDIT_POLICY_DIR, filename)
        try:
            if filename.endswith('.txt'):
                loader = TextLoader(filepath, encoding='utf-8')
                documents.extend(loader.load())
            # If you add PDF policies, uncomment this line (requires `!pip install pypdf` or `!pip install PyMuPDF`)
            # elif filename.endswith('.pdf'):
            #     from langchain_community.document_loaders import PyPDFLoader # Import here if only used conditionally
            #     loader = PyPDFLoader(filepath)
            #     documents.extend(loader.load())
            else:
                print(f"Skipping unsupported file type: {filename}")
        except Exception as e:
            print(f"Error loading document {filename}: {e}")
            continue

    if not documents:
        print("No policy documents found. RAG will not function. Please ensure policy files are in 'credit_policy_documents'.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(texts)} text chunks.")

    embeddings = OpenAIEmbeddings() # Will pick up key from os.environ
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH) # Save the index for future use
    print(f"RAG knowledge base created and saved to {VECTOR_DB_PATH}.")
    return vectorstore

# --- GenAI Module: Policy Compliance Checker ---
def check_policy_compliance_rag(vectorstore, loan_application_details):
    """
    Uses RAG to find relevant policy clauses and then an LLM to assess compliance.
    `loan_application_details` is a dictionary of structured loan data for a single applicant.
    """
    if not vectorstore:
        return {
            "policy_compliance_status": "Skipped - No Policy Knowledge Base",
            "relevant_policy_citations": [],
            "compliance_discrepancies": "N/A"
        }

    # Format the loan details into a natural language query for the RAG system
    query_text = (
        f"Loan Application Summary for ID {loan_application_details['loan_id']}:\n"
        f"Income: ${loan_application_details['income']:,}\n"
        f"Age: {loan_application_details['age']} years\n"
        f"Loan Amount: ${loan_application_details['loan_amount']:,}\n"
        f"Credit Score: {loan_application_details['credit_score']}\n"
        f"Debt-to-Income Ratio (DTI): {loan_application_details['debt_to_income_ratio']:.1f}%\n"
        f"Employment Duration: {loan_application_details['employment_duration_years']} years\n"
        f"Purpose of Loan: {loan_application_details['purpose_of_loan_simplified']}\n"
        f"Collateral Provided: {loan_application_details['collateral_provided']}\n\n"
        "Please assess this application against our credit policies and provide a compliance summary."
    )

    # Define the prompt for the LLM to assess compliance based on retrieved context
    compliance_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an AI financial compliance officer. Your task is to rigorously assess a loan application against the provided credit policy clauses.
        Your analysis must be precise. If the application does not meet a policy requirement, state the non-compliance clearly and cite the exact policy section.
        If it requires special review/exception based on the policies, indicate "Requires Review". Otherwise, state "Compliant".

        Output must be strictly JSON, adhering to the following schema:
        {{
          "policy_compliance_status": "string (e.g., 'Compliant', 'Non-Compliant - High DTI Exceeds Policy 1.2', 'Requires Review - Employment Stability')",
          "relevant_policy_citations": "list of strings (e.g., 'policy_01_general_lending.txt, Section 1.2 DTI')",
          "compliance_discrepancies": "string (concise explanation of the non-compliance or reason for review, 'N/A' if compliant)"
        }}
        Ensure the 'policy_compliance_status' field starts with 'Compliant', 'Non-Compliant', or 'Requires Review' followed by specific details if applicable.
        Prioritize identifying non-compliance first. Provide clear and specific policy references if possible.
        """),
        MessagesPlaceholder(variable_name="context"), # This is where retrieved documents go
        HumanMessage(content="{input}") # This maps to the "input" in retrieval_chain.invoke
    ])

    # Initialize the LLM for generation
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0) # Will pick up key from os.environ

    # Create the RAG chain
    document_chain = create_stuff_documents_chain(llm, compliance_prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 5}), document_chain)

    try:
        # Invoke the chain with the formatted query
        response = retrieval_chain.invoke({"input": query_text})

        raw_llm_output = response['answer']
        retrieved_docs = response['context'] # Access retrieved documents

        # Attempt to parse the answer, assuming it's JSON
        try:
            compliance_output = json.loads(raw_llm_output)
        except json.JSONDecodeError:
            print(f"Warning: LLM did not return valid JSON for Loan ID {loan_application_details['loan_id']}. Raw output: {raw_llm_output[:200]}...")
            compliance_output = {
                "policy_compliance_status": "Parsing Error - Manual Review Needed",
                "relevant_policy_citations": [],
                "compliance_discrepancies": f"LLM output not valid JSON: {raw_llm_output[:100]}..."
            }

        # Extract citations from the retrieved documents' metadata and format them nicely
        citations = []
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'Unknown Policy')
            # Extract filename (e.g., 'policy_01_general_lending.txt')
            filename_only = os.path.basename(source)
            # Try to find a section reference in the document content
            section_match = re.search(r'Section \d+\.\d+', doc.page_content)
            if section_match:
                citations.append(f"{filename_only}, {section_match.group(0)}")
            else:
                citations.append(filename_only)

        # Use the citations from the LLM's response if it provided them, otherwise use our extracted ones
        if not compliance_output.get('relevant_policy_citations'):
            compliance_output['relevant_policy_citations'] = sorted(list(set(citations))) # Deduplicate and sort

        return compliance_output

    except Exception as e:
        print(f"Error during policy compliance check for Loan ID {loan_application_details['loan_id']}: {e}")
        return {
            "policy_compliance_status": "Error - Manual Review Needed",
            "relevant_policy_citations": [],
            "compliance_discrepancies": str(e)
        }

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Setup RAG Knowledge Base
    credit_policy_vectorstore = setup_rag_knowledge_base()
    if not credit_policy_vectorstore:
        print("Failed to setup RAG knowledge base. Exiting.")
        exit()

    # 2. Load Structured Loan Application Data
    print(f"\n--- Loading loan applications from {LOAN_APPLICATIONS_FILE} ---")
    try:
        df_loans = pd.read_csv(LOAN_APPLICATIONS_FILE)
    except FileNotFoundError:
        print(f"Error: {LOAN_APPLICATIONS_FILE} not found. Please ensure it's uploaded to your Colab session.")
        exit()

    # Prepare list to store results
    compliance_results_list = []

    print(f"\n--- Starting RAG-Powered Policy Compliance Check for Each Loan ---")
    for index, row in df_loans.iterrows():
        loan_id = str(row['loan_id'])
        os.environ['CURRENT_LOAN_ID'] = loan_id # For logging/debugging

        print(f"\n--- Checking Loan ID: {loan_id} ({index + 1}/{len(df_loans)}) ---")
        loan_details = row.to_dict()

        # Run the RAG-powered compliance check
        compliance_output = check_policy_compliance_rag(credit_policy_vectorstore, loan_details)

        # Combine original loan details with compliance results
        loan_details.update(compliance_output)
        compliance_results_list.append(loan_details)

    # Convert results to a DataFrame
    df_compliance_report = pd.DataFrame(compliance_results_list)

    print("\n--- Final Loan Compliance Report ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_compliance_report[[
        'loan_id', 'income', 'credit_score', 'debt_to_income_ratio', 'employment_duration_years',
        'policy_compliance_status', 'compliance_discrepancies', 'relevant_policy_citations'
    ]].to_string())

    print("\n--- Why this is impressive for your resume ---")
    print("This project demonstrates:")
    print("1.  **Direct Compliance Automation:** Automated checking of loan applications against complex, internal credit policies using GenAI and RAG.")
    print("2.  **Reduced Manual Effort & Error:** Significantly decreases the time and human error involved in manual policy review.")
    print("3.  **Auditability & Transparency:** Provides clear, machine-generated explanations and policy citations for compliance decisions.")
    print("4.  **Strategic Business Impact:** Directly addresses regulatory risk, improves operational efficiency, and standardizes decision-making in financial institutions.")
    print("5.  **Advanced GenAI Application:** Shows practical implementation of Retrieval-Augmented Generation for critical enterprise knowledge.")

    # Clean up the temporary environment variable
    if 'CURRENT_LOAN_ID' in os.environ:
        del os.environ['CURRENT_LOAN_ID']