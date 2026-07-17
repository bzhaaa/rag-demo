from typing import List, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunking.contracts import ChunkingStrategy, ChunkRecord, ParsedBlock
from app.config import Settings


class LegacyRecursiveChunkingStrategy:
    name = "legacy_recursive"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def chunk(self, blocks: List[ParsedBlock]) -> List[ChunkRecord]:
        records: List[ChunkRecord] = []
        for block in blocks:
            for text in self.splitter.split_text(block.text):
                records.append(
                    ChunkRecord(
                        content=text,
                        parent_content=text,
                        parent_index=len(records),
                        section_title=block.section_title,
                        section_path=block.section_path,
                        page_start=block.page_start,
                        page_end=block.page_end,
                        chunking_strategy=self.name,
                        chunking_version=self.settings.chunking_version,
                    )
                )
        return records


class ParentChildChunkingStrategy:
    name = "structure_parent_child"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_parent_size,
            chunk_overlap=settings.chunk_parent_overlap,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def chunk(self, blocks: List[ParsedBlock]) -> List[ChunkRecord]:
        records: List[ChunkRecord] = []
        for parent_index, parent in enumerate(self._build_parents(blocks)):
            header = " > ".join(parent.section_path)
            parent_content = self._with_header(
                parent.text,
                header,
            )
            for child in self._build_children(parent):
                content = child.text
                if self.settings.chunk_context_header_enabled:
                    content = self._with_header(
                        content,
                        header,
                    )
                records.append(
                    ChunkRecord(
                        content=content,
                        parent_content=parent_content,
                        parent_index=parent_index,
                        section_title=parent.section_title,
                        section_path=parent.section_path,
                        page_start=child.page_start,
                        page_end=child.page_end,
                        chunking_strategy=self.name,
                        chunking_version=self.settings.chunking_version,
                    )
                )
        return records

    def _build_parents(self, blocks: Sequence[ParsedBlock]) -> List[ParsedBlock]:
        parents: List[ParsedBlock] = []
        section_blocks: List[ParsedBlock] = []
        for block in blocks:
            if not block.text.strip():
                continue
            if (
                section_blocks
                and block.section_path != section_blocks[-1].section_path
            ):
                parents.extend(self._parents_for_section(section_blocks))
                section_blocks = []
            section_blocks.append(block)
        if section_blocks:
            parents.extend(self._parents_for_section(section_blocks))
        return self._merge_short_parents(parents)

    def _parents_for_section(
        self, blocks: Sequence[ParsedBlock]
    ) -> List[ParsedBlock]:
        parents: List[ParsedBlock] = []
        current: List[ParsedBlock] = []
        for block in blocks:
            if (
                current
                and self._combined_length(current, block)
                > self.settings.chunk_parent_size
            ):
                parents.extend(self._flush_parent(current))
                current = self._tail_overlap(
                    current,
                    self.settings.chunk_parent_overlap,
                )
                while (
                    current
                    and self._combined_length(current, block)
                    > self.settings.chunk_parent_size
                ):
                    current.pop(0)
            current.append(block)
        if current:
            parents.extend(self._flush_parent(current))
        return parents

    def _flush_parent(self, blocks: Sequence[ParsedBlock]) -> List[ParsedBlock]:
        text = "\n\n".join(block.text for block in blocks if block.text.strip())
        if not text:
            return []
        first = blocks[0]
        last = blocks[-1]
        if len(text) <= self.settings.chunk_parent_size:
            return [
                ParsedBlock(
                    text=text,
                    block_type="parent",
                    section_title=last.section_title or first.section_title,
                    section_path=list(last.section_path or first.section_path),
                    page_start=first.page_start,
                    page_end=last.page_end,
                    atomic=len(blocks) == 1 and first.atomic,
                    parts=list(blocks),
                )
            ]
        return [
            ParsedBlock(
                text=piece,
                block_type="parent",
                section_title=first.section_title,
                section_path=list(first.section_path),
                page_start=first.page_start,
                page_end=last.page_end,
                parts=[
                    ParsedBlock(
                        text=piece,
                        block_type=first.block_type,
                        section_title=first.section_title,
                        section_path=list(first.section_path),
                        page_start=first.page_start,
                        page_end=last.page_end,
                    )
                ],
            )
            for piece in self.parent_splitter.split_text(text)
            if piece.strip()
        ]

    def _merge_short_parents(
        self, parents: List[ParsedBlock]
    ) -> List[ParsedBlock]:
        result: List[ParsedBlock] = []
        for parent in parents:
            if (
                result
                and len(parent.text) < self.settings.chunk_min_size
                and result[-1].section_path == parent.section_path
                and len(result[-1].text) + len(parent.text) + 2
                <= self.settings.chunk_parent_size
            ):
                result[-1].text = f"{result[-1].text}\n\n{parent.text}"
                result[-1].page_end = parent.page_end
                result[-1].parts.extend(parent.parts)
            else:
                result.append(parent)
        return result

    def _build_children(self, parent: ParsedBlock) -> List[ParsedBlock]:
        fragments: List[ParsedBlock] = []
        for part in parent.parts or [parent]:
            texts = (
                [part.text]
                if part.atomic and len(part.text) <= self.settings.chunk_size
                else self.child_splitter.split_text(part.text)
            )
            fragments.extend(
                ParsedBlock(
                    text=text,
                    block_type=part.block_type,
                    section_title=part.section_title,
                    section_path=list(part.section_path),
                    page_start=part.page_start,
                    page_end=part.page_end,
                    atomic=part.atomic,
                )
                for text in texts
                if text.strip()
            )

        children: List[ParsedBlock] = []
        current: List[ParsedBlock] = []
        for fragment in fragments:
            if (
                current
                and self._combined_length(current, fragment)
                > self.settings.chunk_size
            ):
                children.append(self._join_fragments(current))
                current = self._tail_overlap(
                    current,
                    self.settings.chunk_overlap,
                )
                while (
                    current
                    and self._combined_length(current, fragment)
                    > self.settings.chunk_size
                ):
                    current.pop(0)
            current.append(fragment)
        if current:
            children.append(self._join_fragments(current))
        return self._merge_short_children(children)

    def _merge_short_children(
        self, children: List[ParsedBlock]
    ) -> List[ParsedBlock]:
        result: List[ParsedBlock] = []
        for child in children:
            if (
                result
                and len(child.text) < self.settings.chunk_min_size
                and len(result[-1].text) + len(child.text) + 2
                <= self.settings.chunk_size
            ):
                result[-1].text = f"{result[-1].text}\n\n{child.text}"
                result[-1].page_end = child.page_end
            else:
                result.append(child)
        return result

    @staticmethod
    def _join_fragments(fragments: Sequence[ParsedBlock]) -> ParsedBlock:
        return ParsedBlock(
            text="\n\n".join(fragment.text for fragment in fragments),
            block_type="child",
            section_title=fragments[-1].section_title,
            section_path=list(fragments[-1].section_path),
            page_start=fragments[0].page_start,
            page_end=fragments[-1].page_end,
        )

    @staticmethod
    def _tail_overlap(
        blocks: Sequence[ParsedBlock],
        overlap: int,
    ) -> List[ParsedBlock]:
        if overlap <= 0:
            return []
        selected: List[ParsedBlock] = []
        size = 0
        for block in reversed(blocks):
            block_size = len(block.text) + (2 if selected else 0)
            if selected and size + block_size > overlap:
                break
            if not selected and len(block.text) > overlap:
                return [
                    ParsedBlock(
                        text=block.text[-overlap:],
                        block_type=block.block_type,
                        section_title=block.section_title,
                        section_path=list(block.section_path),
                        page_start=block.page_start,
                        page_end=block.page_end,
                        atomic=False,
                    )
                ]
            selected.append(block)
            size += block_size
            if size >= overlap:
                break
        return list(reversed(selected))

    @staticmethod
    def _combined_length(
        blocks: Sequence[ParsedBlock],
        next_block: ParsedBlock,
    ) -> int:
        return sum(len(block.text) for block in blocks) + (
            2 * len(blocks)
        ) + len(next_block.text)

    @staticmethod
    def _with_header(text: str, header: str) -> str:
        prefix = f"{header}\n"
        if not header or text.startswith(prefix):
            return text
        return f"{prefix}{text}"


def create_chunking_strategy(settings: Settings) -> ChunkingStrategy:
    if settings.chunking_strategy == "legacy_recursive":
        return LegacyRecursiveChunkingStrategy(settings)
    return ParentChildChunkingStrategy(settings)
