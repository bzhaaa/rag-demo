from types import SimpleNamespace

import pytest

from app.chunking import (
    LegacyRecursiveChunkingStrategy,
    ParentChildChunkingStrategy,
    ParsedBlock,
    StructureAwareDocumentParser,
)
from app.config import Settings
from app.ingestion import build_chunks


def chunking_settings(**overrides):
    values = {
        "_env_file": None,
        "chunking_strategy": "structure_parent_child",
        "chunk_size": 120,
        "chunk_overlap": 20,
        "chunk_parent_size": 300,
        "chunk_parent_overlap": 30,
        "chunk_min_size": 20,
        "chunk_context_header_enabled": True,
        "chunking_version": "v2",
        "web_search_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)


def test_markdown_parser_preserves_paths_and_atomic_blocks():
    text = """# 员工制度

## 请假管理

### 审批流程

员工应提前发起申请。

- 直属主管审批
- 人事部门备案

| 角色 | 时限 |
| --- | --- |
| 主管 | 1 天 |

```python
print("leave")
```
"""

    blocks = StructureAwareDocumentParser().parse(
        text.encode("utf-8"),
        "text/markdown",
    )

    assert [item.block_type for item in blocks] == [
        "paragraph",
        "list",
        "table",
        "code",
    ]
    assert all(
        item.section_path == ["员工制度", "请假管理", "审批流程"]
        for item in blocks
    )
    assert all(item.atomic for item in blocks[1:])


def test_plain_text_parser_recognizes_chinese_and_numbered_headings():
    text = """第一章 总则
本制度适用于全体员工。

第一条 适用范围
所有正式员工均适用。

1.1 审批要求
请假必须完成审批。
"""

    blocks = StructureAwareDocumentParser().parse(
        text.encode("utf-8"),
        "text/plain",
    )

    assert blocks[0].section_path == ["第一章 总则"]
    assert blocks[1].section_path == ["第一章 总则", "第一条 适用范围"]
    assert blocks[2].section_path == [
        "第一章 总则",
        "1.1 审批要求",
    ]


def test_pdf_parser_removes_repeated_margins_and_joins_cross_page_text(
    monkeypatch,
):
    page_texts = [
        "公司制度\n第一章 总则\n跨页段落尚未结束\n内部资料",
        "公司制度\n继续内容。\n第二章 其他\n独立内容。\n内部资料",
    ]

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakeReader:
        def __init__(self, _):
            self.pages = [FakePage(text) for text in page_texts]

    monkeypatch.setattr("app.chunking.parsers.PdfReader", FakeReader)

    blocks = StructureAwareDocumentParser().parse(
        b"%PDF-fake",
        "application/pdf",
    )

    assert blocks[0].text == "跨页段落尚未结束 继续内容。"
    assert blocks[0].page_start == 1
    assert blocks[0].page_end == 2
    assert blocks[0].section_path == ["第一章 总则"]
    assert blocks[1].section_path == ["第二章 其他"]
    assert all("公司制度" not in item.text for item in blocks)
    assert all("内部资料" not in item.text for item in blocks)


def test_parent_child_strategy_keeps_parent_context_and_child_bounds():
    blocks = [
        ParsedBlock(
            text=("审批流程要求完整填写申请表。" * 30),
            section_title="审批流程",
            section_path=["员工制度", "请假管理", "审批流程"],
            page_start=1,
            page_end=2,
        )
    ]
    settings = chunking_settings(
        chunk_size=100,
        chunk_overlap=15,
        chunk_parent_size=240,
        chunk_parent_overlap=20,
        chunk_context_header_enabled=False,
    )

    records = ParentChildChunkingStrategy(settings).chunk(blocks)

    assert len(records) > 1
    assert all(len(item.content) <= settings.chunk_size for item in records)
    assert all(
        len(item.parent_content.split("\n", 1)[-1])
        <= settings.chunk_parent_size
        for item in records
    )
    assert all(item.section_title == "审批流程" for item in records)
    assert all(item.page_start == 1 and item.page_end == 2 for item in records)


def test_parent_child_strategy_merges_short_same_section_fragments():
    blocks = [
        ParsedBlock(
            text="主要内容足够长，能够接收后续短片段。",
            section_path=["制度", "范围"],
        ),
        ParsedBlock(
            text="补充。",
            section_path=["制度", "范围"],
        ),
    ]
    settings = chunking_settings(
        chunk_size=100,
        chunk_overlap=10,
        chunk_parent_size=200,
        chunk_parent_overlap=10,
        chunk_min_size=20,
        chunk_context_header_enabled=False,
    )

    records = ParentChildChunkingStrategy(settings).chunk(blocks)

    assert len(records) == 1
    assert "补充。" in records[0].content


def test_legacy_strategy_remains_available():
    records = LegacyRecursiveChunkingStrategy(
        chunking_settings(
            chunking_strategy="legacy_recursive",
            chunk_context_header_enabled=False,
        )
    ).chunk([ParsedBlock(text="legacy text", page_start=3, page_end=3)])

    assert records[0].content == "legacy text"
    assert records[0].parent_content == "legacy text"
    assert records[0].chunking_strategy == "legacy_recursive"


def test_build_chunks_produces_stable_parent_and_child_ids(monkeypatch):
    settings = chunking_settings(
        chunk_size=80,
        chunk_overlap=10,
        chunk_parent_size=160,
        chunk_parent_overlap=20,
        chunk_context_header_enabled=False,
    )
    monkeypatch.setattr("app.ingestion.get_settings", lambda: settings)
    document = SimpleNamespace(
        uuid="doc-uuid",
        visibility="department",
        department=SimpleNamespace(uuid="department-uuid"),
    )
    version = SimpleNamespace(
        uuid="version-uuid",
        version_number=2,
        source_name="policy.md",
    )
    blocks = [
        ParsedBlock(
            text="A" * 220,
            section_title="范围",
            section_path=["制度", "范围"],
            page_start=1,
            page_end=2,
        ).to_mapping()
    ]

    first = build_chunks(document, version, blocks)
    second = build_chunks(document, version, blocks)

    assert [item["chunk_id"] for item in first] == [
        item["chunk_id"] for item in second
    ]
    assert [item["parent_chunk_id"] for item in first] == [
        item["parent_chunk_id"] for item in second
    ]
    assert first[0]["chunk_id"] == "doc-uuid:2:0"
    assert first[0]["parent_chunk_id"] == "doc-uuid:2:parent:0"
    assert first[0]["page_number"] == first[0]["page_start"] == 1
    assert first[0]["chunking_version"] == "v2"


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"chunking_strategy": "fixed"}, "chunking_strategy"),
        ({"chunk_size": 0}, "chunk sizes"),
        ({"chunk_overlap": -1}, "chunk overlaps"),
        ({"chunk_size": 100, "chunk_overlap": 100}, "chunk_overlap"),
        (
            {"chunk_parent_size": 100, "chunk_parent_overlap": 100},
            "chunk_parent_overlap",
        ),
    ],
)
def test_chunking_settings_validation(values, message):
    with pytest.raises(ValueError, match=message):
        chunking_settings(**values)
