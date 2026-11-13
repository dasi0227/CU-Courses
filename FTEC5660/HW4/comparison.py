from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4


VARIANT = "V4"

ORIGINAL_TABLE_TO_TEXT = hw4.table_to_text
ORIGINAL_CHUNK_DOCUMENT = hw4.chunk_document
ORIGINAL_PARSE_QUERY = hw4.parse_query
ORIGINAL_BUILD_INDEX = hw4.build_index
ORIGINAL_RETRIEVE = hw4.retrieve

def table_to_text(table_html: str) -> str:
    if VARIANT in ("V1", "V2", "V3", "V4"):
        return ORIGINAL_TABLE_TO_TEXT(table_html)
    return BeautifulSoup(table_html, "lxml").get_text("\n", strip=True)

def chunk_document(file_name: str, html: str) -> list[Document]:
    if VARIANT in ("V1", "V2", "V3", "V4"):
        return ORIGINAL_CHUNK_DOCUMENT(file_name, html)

    soup = BeautifulSoup(html, "lxml")

    key = soup.title.get_text(" ", strip=True)
    company = re.sub(r"\s*\([^)]*\)\s*$", "", soup.h1.get_text(" ", strip=True)).strip()
    year = re.search(r"(?:annual report\s+|full year\s+|fy\s*)(\d{4})", key, re.I)
    if year is None:
        raise ValueError(f"Missing fiscal year: {key}")
    if "annual report" in key.lower():
        doc_type = "annual_report"
    elif "earnings call" in key.lower() or "earnings conference call" in key.lower():
        doc_type = "earnings_call"
    elif "results review" in key.lower():
        doc_type = "research_report"
    else:
        raise ValueError(f"Unknown document type: {key}")
    
    metadata = {
        "source": file_name,
        "company": company,
        "fiscal_year": int(year.group(1)),
        "doc_type": doc_type,
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = hw4.MAX_CHUNK_CHARS,
        chunk_overlap = 100,
        separators = ["\n\n", "\n", " ", ""],
    )
    chunks = []
    for text in splitter.split_text(soup.get_text("\n", strip=True)):
        chunks.append(Document(page_content=text, metadata=metadata.copy()))
    
    return chunks


def parse_query(question: str) -> dict:
    if VARIANT in ("V3", "V4"):
        return ORIGINAL_PARSE_QUERY(question)
    return {}


def build_index(chunks: list[Document]) -> Any:
    if VARIANT in ("V2", "V3", "V4"):
        return ORIGINAL_BUILD_INDEX(chunks)

    chunk_tokens = [chunk.page_content.lower().split() for chunk in chunks] if VARIANT in ("B0",) else [re.findall(r"[a-z]+|\d+(?:[.,]\d+)*", chunk.page_content.lower()) for chunk in chunks]
    
    return {
        "chunks": chunks,
        "bm25_a": BM25Okapi(chunk_tokens) if chunk_tokens else None,
    }


def retrieve(index: Any, question: str) -> list[Document]:
    if VARIANT in ("V4",):
        return ORIGINAL_RETRIEVE(index, question)
    
    chunks = index["chunks"]
    if not chunks:
        return []

    question_tokens = question.lower().split() if VARIANT in ("B0",) else re.findall(r"[a-z]+|\d+(?:[.,]\d+)*", question.lower())

    if VARIANT in ("B0", "V1", "V2"):
        scores_a = index["bm25_a"].get_scores(question_tokens)
        candidates_a = sorted(range(len(chunks)), key=lambda i: scores_a[i], reverse=True)[:hw4.MAX_RETRIEVED]
        return [chunks[i] for i in candidates_a]

    question_metadata = parse_query(question)
    companies = {hw4.company_key(company) for company in question_metadata["companies"]}
    if not companies.issubset(index["companies"]):
        return []
    candidates_b = []
    for i, chunk in enumerate(chunks):
        if hw4.company_key(chunk.metadata["company"]) in companies and chunk.metadata["doc_type"] == question_metadata["doc_type"]:
            candidates_b.append(i)
    scores_b = index["bm25_b"].get_scores(question_tokens)
    candidates_b = sorted(candidates_b, key=lambda i: scores_b[i], reverse=True)[:hw4.MAX_RETRIEVED]
    return [chunks[i] for i in candidates_b]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FTEC5660 HW4 configurations")
    parser.add_argument("variant", choices=("B0", "V1", "V2", "V3", "V4"))
    args = parser.parse_args()
    VARIANT = args.variant

    hw4.table_to_text = table_to_text
    hw4.chunk_document = chunk_document
    hw4.parse_query = parse_query
    hw4.build_index = build_index
    hw4.retrieve = retrieve

    folder = Path(__file__).resolve().parent
    sys.argv = [sys.argv[0], "--data-folder", str(folder / "public_test"), "--output", str(folder / f"results_{VARIANT}.csv")]
    hw4.main()
