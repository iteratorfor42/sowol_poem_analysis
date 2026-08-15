"""
visualize.py
------------
시별 감정 분석 결과를 시각화.

1) 시 x 감정 히트맵: 9편 전체를 한눈에 비교
2) 시별 top-N 감정 막대그래프

세션 소개에서 언급된 "텍스트를 벡터 공간에 투영해 비교"라는 접근 축을,
임베딩 기반 UMAP/t-SNE 대신 KPoEM의 43차원 감정 확률 벡터로 구현한 버전.
감정 라벨 자체가 이미 해석 가능한 축이라, 별도의 차원 축소 없이도
시들 사이의 정서적 근접성/거리를 히트맵으로 바로 읽을 수 있다는 장점이 있다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from analyze import PoemRecord
from emotion_model import KPoEMClassifier


def _configure_korean_font() -> None:
    """
    한글 폰트가 없으면 그래프의 한글 라벨이 네모(□)로 깨진다.
    시스템에 설치된 한글 지원 폰트를 찾아 matplotlib에 등록한다.
    (못 찾으면 기본 폰트로 진행하고, 라벨이 깨지면 폰트를 별도 설치해야 함)
    """
    candidates = ["NanumGothic", "AppleGothic", "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def build_emotion_matrix(
    records: list[PoemRecord], top_n_emotions: int = 12
) -> tuple[list[str], list[str], np.ndarray]:
    """
    (시 제목 목록, 감정 라벨 목록, 확률 행렬)을 반환.
    감정 라벨은 전체 시집 기준으로 평균 확률이 높은 상위 top_n_emotions개만 사용
    (43개를 다 그리면 히트맵이 과밀해짐).
    """
    all_labels = KPoEMClassifier.LABELS
    prob_by_label: dict[str, list[float]] = {label: [] for label in all_labels}

    full_matrix = []
    for record in records:
        emotion_dict = dict(record.poem_level_emotions)
        row = [emotion_dict.get(label, 0.0) for label in all_labels]
        full_matrix.append(row)
        for label, prob in zip(all_labels, row):
            prob_by_label[label].append(prob)

    avg_by_label = {label: sum(v) / len(v) for label, v in prob_by_label.items()}
    top_labels = sorted(avg_by_label, key=avg_by_label.get, reverse=True)[:top_n_emotions]
    label_idx = [all_labels.index(label) for label in top_labels]

    matrix = np.array(full_matrix)[:, label_idx]
    titles = [f"{r.index}. {r.title}" for r in records]
    return titles, top_labels, matrix


def plot_emotion_heatmap(records: list[PoemRecord], out_path: str, top_n_emotions: int = 12) -> None:
    _configure_korean_font()
    titles, labels, matrix = build_emotion_matrix(records, top_n_emotions=top_n_emotions)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), max(5, len(titles) * 0.5)))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles)

    ax.set_title("『진달래꽃』 - 바다가 변하야 뽕나무밭 된다고 (9편) 감정 히트맵")
    fig.colorbar(im, ax=ax, label="감정 확률")
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_emotions_per_poem(records: list[PoemRecord], out_path: str, top_n: int = 5) -> None:
    _configure_korean_font()
    n = len(records)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for ax, record in zip(axes, records):
        top = record.top_emotions(top_n)
        labels = [label for label, _ in top]
        probs = [prob for _, prob in top]
        ax.barh(labels[::-1], probs[::-1], color="#c0392b")
        ax.set_xlim(0, 1)
        ax.set_title(f"{record.index}. {record.title}", fontsize=10)

    for ax in axes[len(records):]:
        ax.axis("off")

    fig.suptitle("시별 상위 감정 (KPoEM)", fontsize=13)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
