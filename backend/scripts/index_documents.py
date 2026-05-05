import os
import glob,logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger=logging.getLogger("indexer")


def index_docs():

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder=os.path.join(cur_dir,"../../backend/data")


    logger.info("="*50)
    try:
        logger.info("Intiliazing mistal ai embeddings model")
        embeddings = MistralAIEmbeddings(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral-embed"
        )

        logger.info("Embedding model intilialized successfully")
    except Exception as e:
        logger.error(f"Failed to initilize embedding model , wrong Mistral API KEY")
        return
    

    
    try:
        logger.info("Logging into Pinecone ... ")
        vector_store=PineconeVectorStore(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            index_name=index_name,
            embedding=embeddings
        )

        logger.info("Pinecone intilialized successfully")
    except Exception as e:
        logger.error(f"Failed to initilize Pinecone , wrong Pinecone API KEY")
        return
    
    pdf_file=glob.glob(os.path.join(data_folder,".pdf"))
    if not pdf_file:
        logger.warning(f"No pdf files")
    logger.info(f"Found {len(pdf_file)} files : {[os.path.basename(f) for f in pdf_file]}")

    all_split=[]
    for pdf_path in pdf_file:
        try:
            logger.info(f"Loading: {os.path.basename(pdf_path)}")
            loader=PyPDFLoader(pdf_path)
            raw_docs=loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100
            )

            splits = text_splitter.split_documents(raw_docs)
            for split in splits:
                split.metadata["source"]=os.path.basename(pdf_path)
            all_split.extend(splits)
            logger.info(f"Split into {len(splits)} chunks")

        except Exception as e:
            logger.error(f"Failed to split document {pdf_path}: {e}")

    if all_split:
        logger.info(f"Uploading chunks to Pinecone")
        try:
            vector_store.add_documents(documents=all_split)
            logger.info("Indexing complete ")
            logger.info(f"Total chunks created: {len(all_split)}")
        except Exception as e:
            logger.error("Failed to upload chunk to pinecone.")
    else:
        logger.error(f"No chhunk found")


if __name__=="__main__":
    index_docs()


