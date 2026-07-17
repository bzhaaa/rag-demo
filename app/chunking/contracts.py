from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class ParsedBlock:
    text: str
    block_type: str = "paragraph"
    section_title: str = ""
    section_path: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    atomic: bool = False
    parts: List["ParsedBlock"] = field(default_factory=list, repr=False)

    @classmethod
    def from_mapping(cls, value: Dict[str, Any]) -> "ParsedBlock":
        page_number = value.get("page_number")
        raw_path = value.get("section_path") or []
        section_path = (
            [item.strip() for item in str(raw_path).split(">") if item.strip()]
            if isinstance(raw_path, str)
            else list(raw_path)
        )
        return cls(
            text=str(value.get("text") or ""),
            block_type=str(value.get("block_type") or "paragraph"),
            section_title=str(value.get("section_title") or ""),
            section_path=section_path,
            page_start=value.get("page_start", page_number),
            page_end=value.get("page_end", page_number),
            atomic=bool(value.get("atomic", False)),
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "block_type": self.block_type,
            "section_title": self.section_title,
            "section_path": list(self.section_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "page_number": self.page_start,
            "atomic": self.atomic,
        }


@dataclass
class ChunkRecord:
    content: str
    parent_content: str
    parent_index: int
    section_title: str = ""
    section_path: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunking_strategy: str = "structure_parent_child"
    chunking_version: str = "v2"


class DocumentParser(Protocol):
    def parse(self, data: bytes, mime_type: str) -> List[ParsedBlock]: ...


class ChunkingStrategy(Protocol):
    name: str

    def chunk(self, blocks: List[ParsedBlock]) -> List[ChunkRecord]: ...
