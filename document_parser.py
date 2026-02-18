import os
from PyPDF2 import PdfReader
from docx import Document

class DocumentParser:
    """Unified parser for multiple document formats."""
    
    def __init__(self):
        """Initialize the document parser."""
        self.supported_formats = {
            'txt': self.parse_txt,
            'pdf': self.parse_pdf,
            'docx': self.parse_docx
        }
    
    def get_file_extension(self, filename):
        return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    def parse_txt(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
    
    def parse_pdf(self, filepath):
        text = []
        
        try:
            # Open PDF file
            reader = PdfReader(filepath)
            
            # Extract text from each page
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            
            # Combine all pages
            full_text = '\n\n'.join(text)
            
            if not full_text.strip():
                raise ValueError("PDF appears to be empty or contains only images")
            
            return full_text
            
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
    
    def parse_docx(self, filepath):
        try:
            # Open DOCX file
            doc = Document(filepath)
            
            # Extract text from all paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
            
            # Combine all paragraphs
            full_text = '\n\n'.join(paragraphs)
            
            if not full_text.strip():
                raise ValueError("DOCX appears to be empty")
            
            return full_text
            
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {str(e)}")
    
    def parse(self, filepath):
        # Get file extension
        filename = os.path.basename(filepath)
        extension = self.get_file_extension(filename)
        
        # Check if format is supported
        if extension not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: .{extension}. "
                f"Supported formats: {', '.join(self.supported_formats.keys())}"
            )
        
        # Parse file using appropriate method
        parser_func = self.supported_formats[extension]
        text = parser_func(filepath)
        
        # Return parsed data with metadata
        return {
            'text': text,
            'filename': filename,
            'format': extension,
            'char_count': len(text),
            'success': True
        }
    
    def is_supported(self, filename):
        extension = self.get_file_extension(filename)
        return extension in self.supported_formats