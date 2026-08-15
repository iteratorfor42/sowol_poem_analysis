"""
main.py
-------
김소월 『진달래꽃』 - "바다가 변하야 뽕나무밭 된다고" 9편에 대해
로드 -> KPoEM 감정 분석 -> 구조화 -> 시각화 를 한 번에 실행하는 CLI.

사용 예:
    python main.py \
        --json data/raw/jindallaekkot_9poems.json \
        --threshold 0.3 \
        --out-json data/output/emotion_analysis.json \
        --out-heatmap data/output/emotion_heatmap.png \
        --out-bar data/output/top_emotions_per_poem.png

김소월(1902~1934)은 저작권 보호기간(사후 70년)이 지난 퍼블릭 도메인
저자라, OCR 프로젝트와 달리 원문 데이터를 저장소에 그대로 포함해도
저작권 문제가 없다. (효경언해 프로젝트의 --ground-truth 옵션처럼
기본값을 꺼두는 식의 별도 조치가 필요 없음)
"""

from __future__ import annotations

import argparse
import json

from preprocess import load_poems
from emotion_model import KPoEMClassifier
from analyze import analyze_collection
from visualize import plot_emotion_heatmap, plot_top_emotions_per_poem


def run_pipeline(
    json_path: str,
    threshold: float,
    analyze_lines: bool,
    out_json: str,
    out_heatmap: str,
    out_bar: str,
    top_n_emotions: int,
):
    meta, poems = load_poems(json_path)
    print(f"{meta['collection']} - {meta['section']} ({meta['author']}) : {len(poems)}편 로드")

    print("KPoEM 모델 로드 중 (최초 실행 시 다운로드로 시간이 걸릴 수 있음)...")
    classifier = KPoEMClassifier()
    print("모델 로드 완료. 분석 시작...")

    records = analyze_collection(
        poems,
        classifier,
        author=meta["author"],
        collection=f"{meta['collection']} - {meta['section']}",
        threshold=threshold,
        analyze_lines=analyze_lines,
    )

    # 구조화된 JSON으로 저장
    output = {
        "collection": meta["collection"],
        "section": meta["section"],
        "author": meta["author"],
        "model": "AKS-DHLAB/KPoEM",
        "threshold": threshold,
        "poems": [record.to_dict() for record in records],
    }
    from pathlib import Path
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"분석 결과 저장 -> {out_json}")

    # 시각화
    plot_emotion_heatmap(records, out_heatmap, top_n_emotions=top_n_emotions)
    print(f"히트맵 저장 -> {out_heatmap}")

    plot_top_emotions_per_poem(records, out_bar)
    print(f"시별 top감정 막대그래프 저장 -> {out_bar}")

    # 콘솔 요약 출력
    print("\n=== 시별 top3 감정 요약 ===")
    for record in records:
        top3 = ", ".join(f"{label}({prob:.2f})" for label, prob in record.top_emotions(3))
        print(f"[{record.index}] {record.title}: {top3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="김소월 『진달래꽃』 9편 KPoEM 감정 분석 파이프라인")
    parser.add_argument("--json", default="data/raw/jindallaekkot_9poems.json", help="시 데이터 JSON 경로")
    parser.add_argument("--threshold", type=float, default=0.3, help="감정 검출 임계값 (기본 0.3)")
    parser.add_argument("--no-lines", action="store_true", help="행 단위 분석을 생략하고 시 전체 단위만 분석")
    parser.add_argument("--out-json", default="data/output/emotion_analysis.json")
    parser.add_argument("--out-heatmap", default="data/output/emotion_heatmap.png")
    parser.add_argument("--out-bar", default="data/output/top_emotions_per_poem.png")
    parser.add_argument("--top-n-emotions", type=int, default=12, help="히트맵에 표시할 감정 개수")
    args = parser.parse_args()

    run_pipeline(
        json_path=args.json,
        threshold=args.threshold,
        analyze_lines=not args.no_lines,
        out_json=args.out_json,
        out_heatmap=args.out_heatmap,
        out_bar=args.out_bar,
        top_n_emotions=args.top_n_emotions,
    )
