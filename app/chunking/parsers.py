import re
from collections import Counter
from io import BytesIO
from typing import List, Optional, Sequence, Tuple

from pypdf import PdfReader

from app.chunking.contracts import ParsedBlock

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CHINESE_HEADING = re.compile(
    r"^\s*(?:"
    r"第[一二三四五六七八九十百千万零〇\d]+[章节条款编部卷]"
    r"|[一二三四五六七八九十]+、"
    r"|\d+(?:\.\d+){0,5}[、.\s]"
    r")\s*(?:.+)?$"
)
LIST_LINE = re.compile(
    r"^\s*(?:[-*+]\s+|\d+[.)、]\s+|[（(][一二三四五六七八九十\d]+[）)]\s*)"
)
TABLE_SEPARATOR = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
SENTENCE_END = re.compile(r"[。！？!?；;：:.\"'”’）)\]】]$")


class StructureAwareDocumentParser:
    def parse(self, data: bytes, mime_type: str) -> List[ParsedBlock]:
        if mime_type == "application/pdf":
            return self._parse_pdf(data)
        text = data.decode("utf-8-sig")
        if not text.strip():
            return []
        if mime_type == "text/markdown":
            return self._parse_markdown(text)
        blocks, _ = self._parse_plain_text(text)
        return blocks

    def _parse_markdown(self, text: str) -> List[ParsedBlock]:
        blocks: List[ParsedBlock] = []
        section_stack: List[str] = []
        paragraph: List[str] = []
        lines = text.replace("\r\n", "\n").splitlines()
        index = 0

        def flush_paragraph() -> None:
            if paragraph:
                blocks.append(
                    self._block(
                        "\n".join(paragraph),
                        "paragraph",
                        section_stack,
                    )
                )
                paragraph.clear()

        while index < len(lines):
            line = lines[index]
            heading = MARKDOWN_HEADING.match(line)
            if heading:
                flush_paragraph()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                section_stack[:] = section_stack[: level - 1]
                section_stack.append(title)
                index += 1
                continue
            if line.strip().startswith("```"):
                flush_paragraph()
                marker = line.strip()[:3]
                fenced = [line]
                index += 1
                while index < len(lines):
                    fenced.append(lines[index])
                    if lines[index].strip().startswith(marker):
                        index += 1
                        break
                    index += 1
                blocks.append(
                    self._block(
                        "\n".join(fenced),
                        "code",
                        section_stack,
                        atomic=True,
                    )
                )
                continue
            if self._is_table_start(lines, index):
                flush_paragraph()
                table = [line, lines[index + 1]]
                index += 2
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    table.append(lines[index])
                    index += 1
                blocks.append(
                    self._block(
                        "\n".join(table),
                        "table",
                        section_stack,
                        atomic=True,
                    )
                )
                continue
            if LIST_LINE.match(line):
                flush_paragraph()
                items = [line]
                index += 1
                while index < len(lines) and (
                    LIST_LINE.match(lines[index])
                    or (
                        lines[index].startswith((" ", "\t"))
                        and bool(lines[index].strip())
                    )
                ):
                    items.append(lines[index])
                    index += 1
                blocks.append(
                    self._block(
                        "\n".join(items),
                        "list",
                        section_stack,
                        atomic=True,
                    )
                )
                continue
            if not line.strip():
                flush_paragraph()
            else:
                paragraph.append(line)
            index += 1
        flush_paragraph()
        return blocks

    def _parse_plain_text(
        self,
        text: str,
        page_number: Optional[int] = None,
        section_stack: Optional[List[str]] = None,
    ) -> Tuple[List[ParsedBlock], List[str]]:
        blocks: List[ParsedBlock] = []
        path = list(section_stack or [])
        groups = re.split(r"\n\s*\n+", text.replace("\r\n", "\n"))
        for group in groups:
            lines = [line.rstrip() for line in group.splitlines() if line.strip()]
            if not lines:
                continue
            first_line = lines[0].strip()
            if CHINESE_HEADING.match(first_line):
                level = self._heading_level(first_line)
                path[:] = path[: level - 1]
                path.append(first_line)
                lines = lines[1:]
                if not lines:
                    continue
            normalized = "\n".join(lines).strip()
            block_type = (
                "list"
                if lines and all(LIST_LINE.match(line) for line in lines)
                else "paragraph"
            )
            blocks.append(
                self._block(
                    normalized,
                    block_type,
                    path,
                    page_number,
                    atomic=block_type == "list",
                )
            )
        return blocks, path

    def _parse_pdf(self, data: bytes) -> List[ParsedBlock]:
        reader = PdfReader(BytesIO(data))
        page_lines = [
            [
                line.strip()
                for line in (page.extract_text() or "").splitlines()
                if line.strip()
            ]
            for page in reader.pages
        ]
        repeated_headers, repeated_footers = self._repeated_margins(page_lines)
        blocks: List[ParsedBlock] = []
        section_stack: List[str] = []
        for page_number, lines in enumerate(page_lines, start=1):
            cleaned = [
                line
                for index, line in enumerate(lines)
                if not (
                    (index < 2 and line in repeated_headers)
                    or (
                        index >= max(0, len(lines) - 2)
                        and line in repeated_footers
                    )
                )
            ]
            page_blocks, section_stack = self._parse_plain_text(
                self._join_pdf_lines(cleaned),
                page_number,
                section_stack,
            )
            if (
                blocks
                and page_blocks
                and self._should_join_across_pages(blocks[-1], page_blocks[0])
            ):
                previous = blocks[-1]
                current = page_blocks.pop(0)
                previous.text = f"{previous.text} {current.text}".strip()
                previous.page_end = current.page_end
            blocks.extend(page_blocks)
        return blocks

    @staticmethod
    def _block(
        text: str,
        block_type: str,
        section_path: Sequence[str],
        page_number: Optional[int] = None,
        atomic: bool = False,
    ) -> ParsedBlock:
        return ParsedBlock(
            text=text.strip(),
            block_type=block_type,
            section_title=section_path[-1] if section_path else "",
            section_path=list(section_path),
            page_start=page_number,
            page_end=page_number,
            atomic=atomic,
        )

    @staticmethod
    def _is_table_start(lines: Sequence[str], index: int) -> bool:
        return (
            index + 1 < len(lines)
            and "|" in lines[index]
            and bool(TABLE_SEPARATOR.match(lines[index + 1]))
        )

    @staticmethod
    def _heading_level(title: str) -> int:
        if re.match(r"^第.+章", title):
            return 1
        if re.match(r"^第.+节", title):
            return 2
        if re.match(r"^第.+条", title):
            return 3
        number = re.match(r"^(\d+(?:\.\d+)*)", title)
        if number:
            return min(6, number.group(1).count(".") + 1)
        return 2

    @staticmethod
    def _join_pdf_lines(lines: Sequence[str]) -> str:
        paragraphs: List[str] = []
        current: List[str] = []
        for line in lines:
            if CHINESE_HEADING.match(line) or LIST_LINE.match(line):
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                paragraphs.append(line)
                continue
            current.append(line)
            if SENTENCE_END.search(line):
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))
        return "\n\n".join(paragraphs)

    @staticmethod
    def _repeated_margins(
        pages: Sequence[Sequence[str]],
    ) -> Tuple[set[str], set[str]]:
        threshold = max(2, (len(pages) + 1) // 2)
        header_counts = Counter(
            line for page in pages for line in page[:2] if len(line) <= 120
        )
        footer_counts = Counter(
            line for page in pages for line in page[-2:] if len(line) <= 120
        )
        return (
            {line for line, count in header_counts.items() if count >= threshold},
            {line for line, count in footer_counts.items() if count >= threshold},
        )

    @staticmethod
    def _should_join_across_pages(
        previous: ParsedBlock,
        current: ParsedBlock,
    ) -> bool:
        first_line = current.text.splitlines()[0] if current.text else ""
        return (
            previous.block_type == "paragraph"
            and current.block_type == "paragraph"
            and previous.section_path == current.section_path
            and not SENTENCE_END.search(previous.text)
            and not CHINESE_HEADING.match(first_line)
            and not LIST_LINE.match(first_line)
        )
