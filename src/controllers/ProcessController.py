from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from models import ProcessingEnum
from typing import List
from dataclasses import dataclass

@dataclass
class Document:
    page_content: str
    metadata: dict

class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):

        file_ext = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(
            self.project_path,
            file_id
        )

        if not os.path.exists(file_path):
            return None

        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")

        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)
        
        return None

    def get_file_content(self, file_id: str):

        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()

        return None

    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int=100, overlap_size: int=20):

        file_content_texts = [
            rec.page_content
            for rec in file_content
        ]

        file_content_metadata = [
            rec.metadata
            for rec in file_content
        ]

        # chunks = text_splitter.create_documents(
        #     file_content_texts,
        #     metadatas=file_content_metadata
        # )

        chunks = self.process_simpler_splitter(
            texts=file_content_texts,
            metadatas=file_content_metadata,
            chunk_size=chunk_size,
        )

        return chunks

    def process_simpler_splitter(self, texts: List[str], metadatas: List[dict], chunk_size: int, splitter_tag: str="\n"):
        
        full_text = " ".join(texts)

        # split by splitter_tag
        lines = [ doc.strip() for doc in full_text.split(splitter_tag) if len(doc.strip()) > 1 ]

        chunks = []
        current_chunk = ""

        for line in lines:
            current_chunk += line + splitter_tag
            if len(current_chunk) >= chunk_size:
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={}
                ))

                current_chunk = ""

        if len(current_chunk) >= 0:
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={}
            ))

        return chunks


    def split_text_by_semantics(self,text: str, max_chunk_size: int = 1000, min_chunk_size: int = 200) -> List[str]:
        """
        Splits a large body of text into smaller, semantically coherent chunks.

        The process attempts to respect paragraph and sentence boundaries 
        while adhering to the specified size limits.

        Args:
            text (str): The full document text to be split.
            max_chunk_size (int): The maximum character size for any single chunk.
            min_chunk_size (int): The minimum acceptable size for a chunk.

        Returns:
            List[str]: A list of semantically grouped text chunks.
        """
        
        # 1. Initial Split by Paragraphs (Best semantic unit)
        # Use regex to split by two or more newlines, which usually denotes a paragraph break.
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        chunks: List[str] = []
        current_chunk_sentences: List[str] = []
        current_size = 0

        for i, paragraph in enumerate(paragraphs):
            # 2. Split the paragraph into individual sentences
            # This regex splits on periods, question marks, or exclamation points 
            # that are followed by whitespace and an uppercase letter, while avoiding 
            # common abbreviations (like "Mr.", "Dr.").
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
            
            # Append the initial paragraphs/sentences to the working chunk
            new_sentences = [s.strip() for s in sentences if s.strip()]

            # If adding all new sentences exceeds the max size, we must cut off.
            while (current_size + sum(len(s) for s in new_sentences) + 3) > max_chunk_size and new_sentences:
                # If the current chunk is not empty and is already above the minimum size, finalize it.
                if current_chunk_sentences and current_size >= min_chunk_size:
                    chunks.append("\n\n".join(current_chunk_sentences))
                    # Reset for the next chunk
                    current_chunk_sentences = []
                    current_size = 0
                    break # Exit the while loop to process the remaining sentences on the next iteration
                
                # If we can't fit the next full paragraph and the current chunk is too small, 
                # we must force the break.
                if current_chunk_sentences and current_size >= min_chunk_size / 2:
                    chunks.append("\n\n".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_size = 0
                    new_sentences.pop(0) # Remove the sentence that forced the break
                    break
                
                # If we can't break, it means we are stuck. Break the while loop.
                break

            # Add the new sentences to the working chunk
            for sentence in new_sentences:
                # Simple heuristic: Add a separator (space/period) between sentences
                sentence_size = len(sentence) + 1
                
                # Check if adding the sentence exceeds the maximum chunk size
                if current_size + sentence_size > max_chunk_size and current_chunk_sentences:
                    # If it exceeds the limit, finalize the current chunk
                    chunks.append("\n\n".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_size = 0
                    # The current sentence becomes the start of the next chunk
                    current_chunk_sentences.append(sentence)
                    current_size = sentence_size
                    break # Exit the inner loop and continue to the next paragraph
                
                # Otherwise, add the sentence and update the size
                current_chunk_sentences.append(sentence)
                current_size += sentence_size
            
            # Handle the edge case where the entire paragraph fits in one chunk
            # (This is usually handled by the loop break, but this is a safety net)

        # 3. Append any remaining sentences in the buffer
        if current_chunk_sentences:
            chunks.append("\n\n".join(current_chunk_sentences))

        return chunks



