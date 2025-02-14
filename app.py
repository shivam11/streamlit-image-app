import os
import streamlit as st
import pymupdf4llm
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
import io
import mimetypes
import uuid
from typing import List, Dict, Any
from typing import Optional

class GoogleDriveHandler:
    def __init__(self):
        self.SCOPES = [
            'https://www.googleapis.com/auth/drive.file',
            #'https://www.googleapis.com/auth/drive.readonly'
        ]
        self.drive_service = self._initialize_drive_service()
        
    def _initialize_drive_service(self):
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', self.SCOPES)
        
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', self.SCOPES)
            creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
                
        return build('drive', 'v3', credentials=creds)

    def list_pdfs_in_folder(self, folder_id):
        results = self.drive_service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf'",
            fields="files(id, name)"
        ).execute()
        return results.get('files', [])

    def download_file(self, file_id):
        request = self.drive_service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)
        
        done = False
        while done is False:
            _, done = downloader.next_chunk()
            
        return file_content.getvalue()

    def find_file_by_name(self, filename: str) -> Optional[str]:
            """
            Find a file in Google Drive by its name and return its viewable link.
            
            Args:
                filename (str): The name of the file to find
                
            Returns:
                Optional[str]: The viewable link if found, None otherwise
            """
            try:
                # Search for the file by name
                results = self.drive_service.files().list(
                    q=f"name='{filename}' and mimeType='application/pdf'",
                    fields="files(id, name)"
                ).execute()
                
                files = results.get('files', [])
                
                if not files:
                    return None
                    
                # Get the first matching file
                file_id = files[0]['id']
                
                # Ensure the file has public read permission
                try:
                    permission = {
                        'type': 'anyone',
                        'role': 'reader'
                    }
                    self.drive_service.permissions().create(
                        fileId=file_id,
                        body=permission
                    ).execute()
                except Exception as e:
                    st.warning(f"Could not update permissions for file {filename}: {str(e)}")
                
                # Return the viewable link
                return f"https://drive.google.com/file/d/{file_id}/view"
                
            except Exception as e:
                st.error(f"Error finding file {filename}: {str(e)}")
                return None

    def upload_file(self, file_data, filename, folder_id=None, mime_type=None):
        file_metadata = {'name': filename}
        if folder_id:
            file_metadata['parents'] = [folder_id]
            
        if mime_type is None:
            mime_type = mimetypes.guess_type(filename)[0]
            if mime_type is None:
                mime_type = 'application/octet-stream'
                
        media = MediaIoBaseUpload(
            io.BytesIO(file_data),
            mimetype=mime_type,
            resumable=True
        )
        
        file = self.drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        self.drive_service.permissions().create(
            fileId=file['id'],
            body=permission
        ).execute()
        
        return f"https://drive.google.com/file/d/{file['id']}/view"

class PDFProcessor:
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, index_name: str, mistral_api_key: str):
        self.drive_handler = GoogleDriveHandler()
        
        # Create temp directory for images
        os.makedirs("temp_images", exist_ok=True)
        st.write("Created temporary directory for images")
        
        # Initialize Mistral client
        self.mistral_client = MistralClient(api_key=mistral_api_key)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            # Parse environment string
            env_parts = pinecone_environment.split('-')
            
            # Handle different cloud providers
            if env_parts[0] == 'gcp':
                cloud = 'gcp'
                if pinecone_environment == 'gcp-starter':
                    region = 'us-west1'  # Default region for starter tier
                else:
                    region = '-'.join(env_parts[1:])  # Rest of the string is region
            elif env_parts[0] == 'aws':
                cloud = 'aws'
                region = '-'.join(env_parts[1:])
            elif env_parts[0] == 'azure':
                cloud = 'azure'
                region = '-'.join(env_parts[1:])
            else:
                raise ValueError(f"Unsupported environment: {pinecone_environment}")

            st.info(f"Creating index with cloud: {cloud}, region: {region}")
            
            # Create new index with Mistral's 1024 dimensions
            self.pc.create_index(
                name=index_name,
                dimension=1024,  # Mistral's embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
        else:
            # Check if existing index has correct dimensions
            index_info = self.pc.describe_index(index_name)
            if index_info.dimension != 1024:
                st.error(f"Existing index '{index_name}' has incorrect dimensions ({index_info.dimension}). Please delete it and run again.")
                raise ValueError(f"Index dimension mismatch: expected 1024, got {index_info.dimension}")
            
            
        self.index = self.pc.Index(index_name)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using Mistral AI."""
        try:
            # Handle batch size limits
            batch_size = 100  # Mistral's recommended batch size
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.mistral_client.embeddings(
                    model="mistral-embed",
                    input=batch  # Changed from 'inputs' to 'input'
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
        except Exception as e:
            st.error(f"Error getting embeddings: {str(e)}")
            return []

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into smaller chunks."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def make_json_serializable(self, obj):  # Added self parameter
        """Convert image metadata to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}  # Added self.
        elif isinstance(obj, list):
            return [self.make_json_serializable(i) for i in obj]  # Added self.
        elif hasattr(obj, '__str__'):  # For Rect and other custom objects
            return str(obj)
        return obj

    def process_pdfs_from_drive(self, source_folder_id: str, image_folder_id: str):
        """Process PDFs and store their embeddings in Pinecone."""
        try:
            pdf_files = self.drive_handler.list_pdfs_in_folder(source_folder_id)
            st.write(f"Found {len(pdf_files)} PDF files in the source folder")

            if not pdf_files:
                st.warning("No PDF files found in the specified folder")
                return

            vectors_to_upsert = []

            for pdf_file in pdf_files:
                st.write(f"Processing {pdf_file['name']}...")
                pdf_content = self.drive_handler.download_file(pdf_file['id'])
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(pdf_content)
                    temp_file_path = temp_file.name
                    
                # Process PDF and extract content
                st.write("Extracting content from PDF...")
                docs = pymupdf4llm.to_markdown(
                    doc=temp_file_path,
                    page_chunks=True,
                    write_images=True,
                    image_path="temp_images",
                    image_format="jpg"
                )
                
                st.write(f"Found {len(docs)} document sections")
                
                for doc_idx, doc in enumerate(docs):
                    # Debug information about images
                    # Process PDF and extract content
                    st.write("Extracting content from PDF...")
                    temp_dir = Path("temp_images")
                    temp_dir.mkdir(exist_ok=True)

                    # Create a dictionary to store image paths
                    image_paths = {}

                    # Modified PDF processing
                    if 'images' in doc:
                        st.write(f"Found {len(doc['images'])} images in section {doc_idx}")
                        processed_images = []
                        
                        for img_idx, img in enumerate(doc['images']):
                            # Generate a unique filename for this image
                            temp_file_name = os.path.basename(doc['metadata']['file_path'])
                            img_filename = f"{temp_dir}/{temp_file_name}-{doc['metadata'].get('page', 0)}-{img_idx}.jpg"
                            img_path = img_filename
                            
                            # Store path in image data
                            img['path'] = str(img_path)
                            st.write(f"Image path: {img['path']}")
                            
                            try:
                                if os.path.exists(img['path']):
                                    with open(img['path'], 'rb') as f:
                                        img_data = f.read()
                                        st.write(f"Uploading {img_filename} to Google Drive...")
                                        img_url = self.drive_handler.upload_file(
                                            img_data,
                                            img_filename,
                                            image_folder_id,
                                            'image/jpeg'
                                        )
                                        processed_images.append({'url': img_url})
                                        st.write(f"Successfully uploaded: {img_url}")
                            except Exception as e:
                                st.error(f"Error processing image {img_path}: {str(e)}")
                        
                        doc['images'] = processed_images
                    else:
                        st.write(f"No images found in section {doc_idx}")


                    # Process text content
                    text_chunks = self.chunk_text(doc['text'])
                    
                    # Get embeddings for text chunks
                    embeddings = self.get_embeddings(text_chunks)
                    
                    # Prepare vectors for Pinecone
                    for chunk, embedding in zip(text_chunks, embeddings):
                        vector = {
                            'id': str(uuid.uuid4()),
                            'values': embedding,
                            'metadata': {
                                'text': chunk,
                                'source': pdf_file['name'],
                                'page': str(doc['metadata'].get('page', '')),
                                'images': json.dumps(self.make_json_serializable(doc.get('images', [])))
                            }
                        }
                        vectors_to_upsert.append(vector)
                
                os.unlink(temp_file_path)

            # Upsert vectors to Pinecone in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)

            st.success(f"Successfully processed {len(vectors_to_upsert)} text chunks!")
            return len(vectors_to_upsert)

        except Exception as e:
            st.error(f"Error processing PDFs: {str(e)}")
            return 0

    def search(self, query: str, top_k: int = 3):
        """Search for similar content using the query."""
        try:
            # Get query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return results.matches
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            return []


    def get_drive_link_for_source(self, source_name: str) -> str:
            """
            Get the Google Drive link for a source document.
            
            Args:
                source_name (str): The name of the source document
                
            Returns:
                str: The Google Drive link or '#' if not found
            """
            drive_link = self.drive_handler.find_file_by_name(source_name)
            return drive_link if drive_link else '#'

    def generate_answer(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        try:
            # Get relevant passages
            results = self.search(query, top_k=top_k)
            
            if not results:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": []
                }
            
            # Prepare context from search results with source IDs
            context_passages = []
            sources = []
            
            for idx, match in enumerate(results, 1):
                source_id = f"source_{idx}"
                passage_text = match.metadata['text']
                source_doc = match.metadata['source']
                page_num = match.metadata.get('page', 'N/A')
                
                # Get the drive link for the source document
                drive_link = self.get_drive_link_for_source(source_doc)
                
                context_passages.append(f"[From {source_doc} (Source {idx}), page {page_num}]: {passage_text}")
                
                source = {
                    "id": source_id,
                    "number": idx,
                    "document": source_doc,
                    "page": page_num,
                    "drive_link": drive_link,
                    "relevance_score": round(float(match.score), 4),
                    "text_snippet": passage_text[:200] + "..." if len(passage_text) > 200 else passage_text
                }
                
                # Add image URLs if available
                try:
                    if 'images' in match.metadata and match.metadata['images']:
                        images = json.loads(match.metadata['images'])
                        if isinstance(images, list):
                            source['images'] = [img.get('url') for img in images if isinstance(img, dict) and 'url' in img]
                except json.JSONDecodeError:
                    st.warning(f"Could not parse images JSON for document {source_doc}")
                
                sources.append(source)
            
            # Join all context passages
            full_context = "\n\n".join(context_passages)
            
            try:
                # Generate answer using Mistral
                messages = [
                    ChatMessage(
                        role="system",
                        content="You are a helpful heating solution expert that provides comprehensive answers based on the provided context. For each statement, include a citation in the format [Source X] where X is the source number. Make sure to include all relevant information from the sources.Write your answer in detail and provide a clear and concise answer. If the information to answer the question is not in the context, say so."
                    ),
                    ChatMessage(
                        role="user",
                        content=f"""Context: {full_context}

    Question: {query}

    Instructions:
    1. Use only information from the provided context
    2. Include citations in the format [Source X] for each piece of information
    3. If the information to answer the question is not in the context, say so
    4. Keep the answer clear and concise"""
                    )
                ]
                
                response = self.mistral_client.chat(
                    model="mistral-large-latest",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )
                
                # Get the generated answer from the response
                generated_answer = response.choices[0].message.content
                
                if not generated_answer:
                    raise ValueError("Empty response from Mistral API")
                    
            except Exception as e:
                st.error(f"Error in answer generation: {str(e)}")
                return {
                    "answer": "Error generating answer from the retrieved content.",
                    "sources": sources
                }
            
            return {
                "answer": generated_answer,
                "sources": sources
            }
            
        except Exception as e:
            st.error(f"Error in generate_answer: {str(e)}")
            return {
                "answer": "An error occurred while generating the answer.",
                "sources": []
            }

def main():
    st.title("PDF Content Extraction and Retrieval with Mistral & Pinecone")
    
    # Initialize configurations
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]
    pinecone_index_name = st.secrets["PINECONE_INDEX_NAME"]
    mistral_api_key = st.secrets["MISTRAL_API_KEY"]
    source_folder_id = st.secrets["SOURCE_FOLDER_ID"]
    image_folder_id = st.secrets["IMAGE_FOLDER_ID"]
    
    processor = PDFProcessor(
        pinecone_api_key=pinecone_api_key,
        pinecone_environment=pinecone_environment,
        index_name=pinecone_index_name,
        mistral_api_key=mistral_api_key
    )
    
    if st.button("Process PDFs from Google Drive"):
        try:
            with st.spinner("Processing PDFs..."):
                st.info(f"Source Folder ID: {source_folder_id}")
                st.info(f"Image Folder ID: {image_folder_id}")
                
                num_chunks = processor.process_pdfs_from_drive(source_folder_id, image_folder_id)
                if num_chunks > 0:
                    st.success(f"Successfully processed {num_chunks} text chunks!")
                else:
                    st.warning("No documents were processed. Please check the source folder.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your folder IDs and permissions.")
    
    # Query interface
    query = st.text_input("Enter your query:")
    if query:
        try:
            with st.spinner("Searching and generating answer..."):
                answer_data = processor.generate_answer(query)
                
                if answer_data["answer"]:
                    st.subheader("Generated Answer:")
                
                # Process the answer to add clickable citations
                answer_text = answer_data["answer"]
                for source in answer_data["sources"]:
                    # Replace [Source X] with clickable links
                    source_ref = f"[Source {source['number']}]"
                    html_link = f'<a href="#{source["id"]}" style="color: #0066cc; text-decoration: underline;">{source_ref}</a>'
                    answer_text = answer_text.replace(source_ref, html_link)
                
                # Display the answer with clickable citations
                st.markdown(answer_text, unsafe_allow_html=True)
                
                if answer_data["sources"]:
                    st.subheader("Sources:")
                    for source in answer_data["sources"]:
                        # Add HTML anchor for source linking
                        st.markdown(f'<div id="{source["id"]}"></div>', unsafe_allow_html=True)
                        
                        # Create the expander with a clickable Drive link
                        with st.expander(
                            f"Source {source['number']}: "
                            f"[{source['document']}]({source.get('drive_link', '#')}) "
                            f"(Page {source['page']})"
                        ):
                            st.markdown(f"""
                            - **Relevance Score**: {source['relevance_score']}
                            - **Text Snippet**: {source['text_snippet']}
                            - **View in Drive**: [{source['document']}]({source.get('drive_link', '#')})
                            """)
                            
                            # Display images if available
                            if source.get('images'):
                                st.subheader("Referenced Images:")
                                for idx, img_url in enumerate(source['images']):
                                    try:
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.image(
                                                img_url,
                                                caption=f"Image {idx + 1}",
                                                use_container_width=True
                                            )
                                        with col2:
                                            st.markdown(f"[View in Drive]({img_url})")
                                    except Exception as img_error:
                                        st.error(f"Error loading image {idx + 1}: {str(img_error)}")
                else:
                    st.warning("No answer could be generated from the available content.")
                    
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.error("Please try rephrasing your question or contact support if the issue persists.")

if __name__ == "__main__":
    main()