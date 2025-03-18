#Requirements-Please make sure you have necessary packages installed in Python before running the code

#importing optional packages

from tqdm import tqdm
import time

#Importing required packages

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer,AutoModelForCausalLM, pipeline

# Step 1 : Defining a function to load the pdf file and convert it into documents

def load_pdf(pdf_path):                #pdf path will be your path to you pdf if the file is located in your systems local directory
    loader=PyPDFLoader(pdf_path)           #Invokes PyPDF function to load and split pdf it into documents/individual "ELEMENTS" with page content and metadata like page number
    pages=loader.load()                    #loader.load() function will
    return pages                           #The pages variable will now hold the data as shown below :
                                           #[Document(content="Text from page 1", metadata={...}),
                                           #Document(content="Text from page 2", metadata={...}),
                                           #Document(content="Text from page 3", metadata={...})]


# Step 2 : Creating "chunks" for efficient storing and retrieval

#Now that we have pdf parsed, we will split each of these pages into numerous "chunks" or batch of words so that it is easy to store and retrieve

def split_text(pages):                                                   #Chunk_size and chunk_overlap are hyperparameters
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    return chunks

'''EXAMPLES- CHUNK_SIZE AND CHUNK_OVERLAPPING
-------------------------------------------------------CHUNK 1

Businesses in the insurance, legal, and healthcare sectors process, sort, and retrieve large volumes of sensitive documents like medical records, financial data, and private data.
Instead of reviewing manually, companies use NLP technology which

--------------------------------------------------------CHUNK 2
allow chat and voice bots to be more human-like when conversing with customers.
Businesses use chatbots to scale customer service capability and quality while keeping operational costs to a minimum.
'''
# If you look at the example above, you'll notice that Chunk 1 and Chunk 2 contain partial parts of a single sentence.
# This can lead to a loss of context and make it difficult for the model to understand the full meaning.
# This is where `chunk_overlap` becomes useful.
# The `chunk_overlap` parameter ensures that a portion of the previous chunk is included in the next chunk.
# This overlapping helps preserve sentence continuity and maintains contextual relevance between chunks.

# Step 4:Defining a function to create a Chroma DB and storing vector embeddings
def create_vector_db(chunks, persist_dir="db"):                                                       #input parameters for the function are the chunks we extracted from pdf and defining the database to store vectors
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")                                 #Invoking Huggingface embedding model
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)     #storing chunks in their vector form to the chroma db that we created earlier
    vectordb.persist()                                                                                #it writes everything to disk so it can be reloaded later with Chroma(persist_directory=...)
    return vectordb                                                                                   #persist_dir: This ensures the vector DB is saved on disk and can be loaded later without recomputing embeddings


# Step 5:Defining a function to  retrieve documents based on user query
def retrieve_context(query, vectordb):                                             #input parameters for retriever function,the query has to be compared against our vectordb to find relevant text
    retriever = vectordb.as_retriever()                                            # Converting the vector database into a retriever for querying relevant documents
    docs = retriever.get_relevant_documents(query)                                 # uses the retriever to search for documents that are semantically similar to the query(user input)
    context = "\n\n".join([doc.page_content for doc in docs])                      #Takes the content (text) from each of the retrieved documents (doc.page_content) and joins them into a single string, separated by double newlines (\n\n).
    return context

'''
---------------------COSINE SIMILARITY------------------------------------------------
Above method uses vector similarity or cosine similarity search to find the most relevant document chunks from the database.
Lets take an example to under stand cosine similarity
Cosine similarity measures the angle between two vectors — it shows how similar two texts (or word vectors) are, regardless of their magnitude (length).

Cosine similarity ranges from -1 to 1:
1 → exactly the same direction (high similarity)
0 → completely different direction (no similarity)
-1 → opposite direction (very rare in NLP)

Cosine Similarity Formula:
cosine_similarity= 𝐴⋅𝐵 / ∥A∥.∥B∥
Where:
A⋅B = dot product of vectors A and B
∥A∥ = magnitude (length) of vector A
∥B∥ = magnitude (length) of vector B

Let’s say you have two words:

"cat" → vector A = [1, 2, 3]
"dog" → vector B = [2, 3, 4]

-----Step 1: Dot product (A·B)------

= (1×2) + (2×3) + (3×4)
= 2 + 6 + 12 = 20

------Step 2: Magnitudes------------

|A| = √(1² + 2² + 3²) = √14 ≈ 3.74
|B| = √(2² + 3² + 4²) = √29 ≈ 5.39

------Step 3: Cosine similarity------

= 20 / (3.74 × 5.39) ≈ 20 / 20.15 ≈ 0.993

Result: High similarity (~0.99) → “cat” and “dog” are semantically close '''

'''
-------------------WHY NOT EUCLIDEAN DISTANCE-----------------------------
#In NLP, we're more interested in the semantic direction of word/sentence vectors, not their absolute magnitude (length).Two sentences can be semantically similar, even if one has more words (longer vector).
#Cosine similarity captures this well because it ignores the vector's length and focuses on how close their meanings are (direction)
--------------------------------------------------------------------------
'''

# Step 6: Defining a function to load  a GPT Model
def load_gpt():
    model_id = "EleutherAI/gpt-neo-1.3B"                       # You can use any model as per your choice(model_id is the ID of the pretrained model you're loading from HuggingFace). EleutherAI/gpt-neo-1.3B is a 1.3 billion parameter GPT-like model, similar to GPT-2/GPT-3 but open-source.
    tokenizer = AutoTokenizer.from_pretrained(model_id)        #The tokenizer splits text into tokens that the model can understand & handles special tokens, padding, and encoding/decoding.
    model = AutoModelForCausalLM.from_pretrained(model_id)     #AutoModelForCausalLM is used for causal language modeling tasks such as text generation,next-word prediction, sentence completion etc.
    return tokenizer, model

# Step 7: Defining a function to generate answer
def generate_answer(query, context, tokenizer, model):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,temperature=0.9, max_length=200, top_p=0.5)
    prompt = f"Answer the following based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    result = pipe(prompt, max_new_tokens=300, do_sample=True)[0]['generated_text']
    return result
'''
Pipeline is a function from the Hugging Face transformers library that helps set up different tasks like text generation, translation, or summarization.
--------------------------------------
"text-generation": This indicates that the pipeline will be used for text generation tasks (e.g., generating an answer or text completion).
--------------------------------------
model=model: The pre-trained model to be used for text generation (e.g., EleutherAI/gpt-neo-1.3B).
--------------------------------------
tokenizer=tokenizer: The tokenizer to encode and decode input and output text (so it can be understood by the model).

--------------------------------------HYPER-PARAMETERS--------------------------------------------------------------------------------------
1. temperature : This controls the randomness or creativity of the generated text.It scales the logits OR output probability distribution) before sampling.
Low temperature (e.g., 0.2): The model will be more deterministic, choosing the most probable next token, which can lead to repetitive, less creative responses.
High temperature (e.g., 1.0): The model will have more freedom in generating text, resulting in more diverse and creative responses.
Range: 0 to 1, with higher values leading to more randomness.
------------------------------------------------------
2. max_length :This is the maximum total length (in terms of tokens) that the model can generate in the output, including both the prompt and the generated text.
If set to 200, the model will generate up to 200 tokens in total (prompt + generated output). If your prompt is 50 tokens, it can generate up to 150 tokens in response.

Lets take an example to understand it better-max_length=200 means the total number of tokens (prompt + output) can't exceed 200.
------------------------------------------------------
3. top_p (Nucleus Sampling)-This hyperparameter controls the probability distribution of the possible next tokens. It’s a form of probabilistic sampling known as nucleus sampling.
Instead of considering all possible tokens, the model selects from the smallest set of tokens whose cumulative probability is less than or equal to top_p.
For example, if top_p=0.5, the model will only consider tokens whose cumulative probability sums up to 50%. This helps focus the model’s output on more likely tokens and avoids choosing random, unlikely tokens.
Range: 0 to 1.
top_p=1: Uses the full probability distribution (no filtering).
top_p=0.9: Limits the token pool to the top 90% probability mass, leading to more diverse but coherent responses.
-------------------------------------------------------
4. max_new_tokens: This controls how many new tokens the model will generate, excluding the input tokens.This is the actual limit on how long the model’s response (generated text) can be.
Example-LET max_new_tokens=300, the model will generate up to 300 tokens after considering the prompt.It ensures that the model doesn't generate excessively long outputs.
------------------------------------------------------
5. do_sample:This controls whether sampling is enabled when generating text. Sampling involves picking the next token based on a probability distribution rather than choosing the most probable token.
If do_sample=True, the model will generate text probabilistically, allowing more variety in responses.
If do_sample=False, the model will always select the most probable next token.
Example:
do_sample=True: Encourages more diverse and creative responses.
do_sample=False: Produces more deterministic, predictable responses.

'''
# Step 8 :Building the main query and invoking functions that we defined earlier

if __name__ == "__main__":          # Main Query invokes all the functions and accepts user input
    pdf_path = r"C:\Users\alexz\Desktop\PHA-Lead-the-Way-Understanding-PHAS.pdf"  # Place your path of the document in your local directory


    for _ in tqdm(range(100), desc="Loading PDF", ncols=100):
        time.sleep(0.05)
    pages = load_pdf(pdf_path)


    for _ in tqdm(range(100), desc="Splitting text into chunks", ncols=100):
        time.sleep(0.05)
    chunks = split_text(pages)


    for _ in tqdm(range(100), desc="Creating vector database", ncols=100):
        time.sleep(0.05)
    vectordb = create_vector_db(chunks)


    for _ in tqdm(range(100), desc="Loading GPT model", ncols=100):
        time.sleep(0.05)
    tokenizer, model = load_gpt()

    while True:
        user_question = input("\nAsk any question from the pdf (or type 'exit'): ")
        if user_question.lower() == 'exit':
            break

        print("\nRetrieving relevant content...")
        context = retrieve_context(user_question, vectordb)

        print("\nGenerating answer...")
        response = generate_answer(user_question, context, tokenizer, model)
        print("\nChatbot Response:\n", response)

    print("\nSession ended.")
    
'''end'''