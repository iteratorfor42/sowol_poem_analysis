"""
preprocess.py
-------------
시 원문 데이터 로드 및 정제.

OCR 프로젝트(ocr_hyogyeong)의 4단계 구조(전처리 -> 분석 -> (감정)인식 -> 후처리)를
텍스트 분석에 맞게 재구성한 것 중 1단계 담당.

효경언해 프로젝트와 다르게, 이번 대상(김소월, 1934년 작고)은 저작권 보호기간
(저작자 사후 70년)이 지난 퍼블릭 도메인 자료라서 원문 자체를
data/raw/ 에 그대로 저장해 두어도 문제가 없다.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Poem:
    index: int
    title: str
    text: str

    @property
    def lines(self) -> list[str]:
        """빈 줄을 제외한 시행 목록."""
        return [line.strip() for line in self.text.split("\n") if line.strip()]


def load_poems(json_path: str) -> tuple[dict, list[Poem]]:
    """
    시집 메타데이터와 개별 시 목록을 로드.
    반환: (collection_meta: dict, poems: list[Poem])
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = {
        "collection": data.get("collection", ""),
        "section": data.get("section", ""),
        "author": data.get("author", ""),
        "note": data.get("note", ""),
    }
    poems = [
        Poem(index=p["index"], title=p["title"], text=clean_text(p["text"]))
        for p in data["poems"]
    ]
    return meta, poems


def clean_text(text: str) -> str:
    """여러 개의 공백/탭을 하나로, 과도한 줄바꿈 정리. 시행 구조(줄바꿈)는 유지."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="시 데이터 로드/정제 확인용")
    parser.add_argument(
        "--json", default="data/raw/jindallaekkot_9poems.json", help="시 데이터 JSON 경로"
    )
    args = parser.parse_args()

    meta, poems = load_poems(args.json)
    print(f"{meta['collection']} - {meta['section']} ({meta['author']})")
    for poem in poems:
        print(f"  [{poem.index}] {poem.title} - {len(poem.lines)}행")
