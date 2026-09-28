from __future__ import annotations

from dataclasses import dataclass, field

TITLE_REGION_PAGES = 2
TITLE_REGION_MAX_CHARS = 2000


@dataclass
class PageSpan:
    page: int
    start: int
    end: int
    sub_document: str | None


@dataclass
class FieldContext:
    full_text: str
    spans: list[PageSpan] = field(default_factory=list)
    title_region_end: int = 0

    @classmethod
    def from_pages(cls, page_order: list[int], page_raw_text: dict[int, str],
                   sub_document_by_page: dict[int, str | None], separator: str = "\n\n") -> "FieldContext":
        spans: list[PageSpan] = []
        parts: list[str] = []
        cursor = 0
        for index, page in enumerate(page_order):
            text = page_raw_text.get(page, "")
            if index:
                cursor += len(separator)
            spans.append(PageSpan(page, cursor, cursor + len(text), sub_document_by_page.get(page)))
            parts.append(text)
            cursor += len(text)
        full_text = separator.join(parts)

        title_pages = [s for s in spans[:TITLE_REGION_PAGES]]
        title_end = min(title_pages[-1].end, TITLE_REGION_MAX_CHARS) if title_pages else 0
        return cls(full_text=full_text, spans=spans, title_region_end=title_end)

    # lookups 
    def page_at(self, offset: int) -> int | None:
        for span in self.spans:
            if span.start <= offset < span.end:
                return span.page
        return None

    def sub_document_at(self, offset: int) -> str | None:
        for span in self.spans:
            if span.start <= offset < span.end:
                return span.sub_document
        return None

    def in_title_region(self, offset: int) -> bool:
        return offset < self.title_region_end

    def title_region(self) -> str:
        return self.full_text[: self.title_region_end]

    def first_sub_document(self) -> str | None:
        for span in self.spans:
            if span.sub_document:
                return span.sub_document
        return None
