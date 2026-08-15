"""
analyze.py
----------
KPoEM 모델로 시 전체/행 단위 감정 분석을 수행하고, 인문학적으로 활용
가능한 구조화된 레코드로 변환한다.

OCR 프로젝트(postprocess.py의 to_structured_record)와 동일한 문제의식:
"분석 결과를 단순 수치 나열이 아니라, 시-행-감정이 연결된 구조화된
데이터로 남겨야 이후 시각화·비교·군집화 등으로 이어질 수 있다."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from preprocess import Poem
from emotion_model import KPoEMClassifier


@dataclass
class LineEmotion:
    line_no: int
    text: str
    emotions: list[tuple[str, float]]


@dataclass
class PoemRecord:
    index: int
    title: str
    author: str
    collection: str
    poem_level_emotions: list[tuple[str, float]]
    lines: list[LineEmotion] = field(default_factory=list)

    def top_emotions(self, n: int = 5) -> list[tuple[str, float]]:
        return self.poem_level_emotions[:n]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "title": self.title,
            "author": self.author,
            "collection": self.collection,
            "poem_level_emotions": [
                {"label": label, "probability": round(prob, 4)}
                for label, prob in self.poem_level_emotions
            ],
            "lines": [
                {
                    "line_no": line.line_no,
                    "text": line.text,
                    "emotions": [
                        {"label": label, "probability": round(prob, 4)}
                        for label, prob in line.emotions
                    ],
                }
                for line in self.lines
            ],
        }


def analyze_poem(
    poem: Poem,
    classifier: KPoEMClassifier,
    author: str,
    collection: str,
    threshold: float = 0.3,
    analyze_lines: bool = True,
) -> PoemRecord:
    """
    시 한 편을 분석. 시 전체 텍스트로 한 번, (옵션) 행 단위로 각각 한 번씩
    KPoEM 모델에 통과시킨다.

    주의: 시행 하나하나는 문맥이 짧아 모델이 학습한 입력 길이(문장~짧은 문단)와
    비교적 잘 맞는 편이지만, 시 전체를 한 번에 넣으면 512 토큰 제한 안에서
    여러 정서가 뭉뚱그려질 수 있다. 두 레벨을 같이 보는 것을 권장.
    """
    poem_level = classifier.analyze(poem.text, threshold=threshold)

    lines: list[LineEmotion] = []
    if analyze_lines:
        for i, line_text in enumerate(poem.lines, start=1):
            line_emotions = classifier.analyze(line_text, threshold=threshold)
            lines.append(LineEmotion(line_no=i, text=line_text, emotions=line_emotions))

    return PoemRecord(
        index=poem.index,
        title=poem.title,
        author=author,
        collection=collection,
        poem_level_emotions=poem_level,
        lines=lines,
    )


def analyze_collection(
    poems: list[Poem],
    classifier: KPoEMClassifier,
    author: str,
    collection: str,
    threshold: float = 0.3,
    analyze_lines: bool = True,
) -> list[PoemRecord]:
    return [
        analyze_poem(
            poem, classifier, author=author, collection=collection,
            threshold=threshold, analyze_lines=analyze_lines,
        )
        for poem in poems
    ]
