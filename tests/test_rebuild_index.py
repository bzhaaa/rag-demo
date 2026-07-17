from types import SimpleNamespace

from scripts import rebuild_hybrid_index


class FakeVectorStore:
    def __init__(self, calls):
        self.calls = calls

    def delete_version(self, version_uuid):
        self.calls.append(("delete", version_uuid))

    def insert_chunks(self, chunks):
        self.calls.append(("insert", list(chunks)))
        return len(chunks)


def test_rebuild_version_deletes_target_version_before_insert(monkeypatch):
    calls = []
    version = SimpleNamespace(
        uuid="version-uuid",
        object_key="documents/policy.md",
        mime_type="text/markdown",
        document=object(),
    )
    monkeypatch.setattr(
        rebuild_hybrid_index,
        "ObjectStorage",
        lambda: SimpleNamespace(download=lambda _: b"# Policy"),
    )
    monkeypatch.setattr(
        rebuild_hybrid_index,
        "parse_document",
        lambda *_: [{"text": "Policy"}],
    )
    monkeypatch.setattr(
        rebuild_hybrid_index,
        "build_chunks",
        lambda *_: [{"chunk_id": "chunk-1"}],
    )
    monkeypatch.setattr(
        rebuild_hybrid_index,
        "MilvusChunkStore",
        lambda: FakeVectorStore(calls),
    )

    inserted = rebuild_hybrid_index.rebuild_version(version)

    assert inserted == 1
    assert calls == [
        ("delete", "version-uuid"),
        ("insert", [{"chunk_id": "chunk-1"}]),
    ]


def test_rebuild_dry_run_does_not_mutate_collection(monkeypatch):
    calls = []
    version = SimpleNamespace(
        uuid="version-uuid",
        object_key="documents/policy.md",
        mime_type="text/markdown",
        document=object(),
    )
    monkeypatch.setattr(
        rebuild_hybrid_index,
        "ObjectStorage",
        lambda: SimpleNamespace(download=lambda _: b"# Policy"),
    )
    monkeypatch.setattr(
        rebuild_hybrid_index,
        "parse_document",
        lambda *_: [{"text": "Policy"}],
    )
    monkeypatch.setattr(
        rebuild_hybrid_index,
        "build_chunks",
        lambda *_: [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}],
    )
    monkeypatch.setattr(
        rebuild_hybrid_index,
        "MilvusChunkStore",
        lambda: FakeVectorStore(calls),
    )

    count = rebuild_hybrid_index.rebuild_version(version, dry_run=True)

    assert count == 2
    assert calls == []
