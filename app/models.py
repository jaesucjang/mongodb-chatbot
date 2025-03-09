from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    text: str

class Query(BaseModel):
    text: str
    use_rag: bool = True