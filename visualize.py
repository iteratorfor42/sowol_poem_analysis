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


def _find_korean_font() -> fm.FontProperties | None:
    """
    한글 폰트를 찾아 FontProperties 객체로 반환한다.

    matplotlib은 여러 언어가 한 파일에 묶인 .ttc(폰트 컬렉션) 안의 특정
    언어 페이스(예: "Noto Sans CJK KR")를 이름만으로는 제대로 찾지 못하는
    경우가 있다. OS 차원(fontconfig)에서는 정상적으로 찾아지는데
    matplotlib의 font.family 이름 검색에서만 실패해서, 폰트가 아예
    없는 것처럼 동작하다가 특정 글자 조합에서만 렌더링이 깨지는 애매한
    상태가 될 수 있다. 그래서 이름 검색이 실패하면 자주 쓰이는 한글 폰트
    파일 경로를 직접 훑어서 파일 경로 기반(FontProperties(fname=...))으로
    폰트를 지정한다 — 이 방식은 이름 등록 여부와 무관하게 항상 동작한다.
    """
    name_candidates = ["NanumGothic", "AppleGothic", "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in name_candidates:
        if name in available:
            return fm.FontProperties(family=name)

    # 이름으로 못 찾았으면, 자주 설치되어 있는 한글 지원 폰트 파일 경로를 직접 탐색
    path_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "C:\\Windows\\Fonts\\malgun.ttf",
        "C:\\Windows\\Fonts\\NanumGothic.ttf",
    ]
    for path in path_candidates:
        if Path(path).exists():
            return fm.FontProperties(fname=path)

    return None


def _configure_korean_font() -> fm.FontProperties | None:
    """
    한글 폰트가 없으면 그래프의 한글 라벨이 네모(□)로 깨진다.
    rcParams(전역 폰트 이름 설정)만으로는 .ttc 폰트에서 실패할 수 있어,
    반환된 FontProperties를 각 텍스트 요소(제목/축 라벨 등)에 개별적으로
    fontproperties=... 인자로 명시 적용하는 것을 권장한다.
    """
    plt.rcParams["axes.unicode_minus"] = False
    font_prop = _find_korean_font()
    if font_prop is not None:
        plt.rcParams["font.family"] = font_prop.get_name()
    return font_prop


def build_emotion_matrix(
    records: list[PoemRecord], top_n_emotions: int = 12) -> tuple[list[str], list[str], np.ndarray]:
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
    font_prop = _configure_korean_font()
    titles, labels, matrix = build_emotion_matrix(records, top_n_emotions=top_n_emotions)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), max(5, len(titles) * 0.5)))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontproperties=font_prop)
    ax.set_yticks(range(len(titles)))
    ax.set_yticklabels(titles, fontproperties=font_prop)

    ax.set_title("『진달래꽃』 - 바다가 변하야 뽕나무밭 된다고 (9편) 감정 히트맵", fontproperties=font_prop)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("감정 확률", fontproperties=font_prop)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_emotions_per_poem(records: list[PoemRecord], out_path: str, top_n: int = 5) -> None:
    font_prop = _configure_korean_font()
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
        ax.set_yticklabels(labels[::-1], fontproperties=font_prop)
        ax.set_xlim(0, 1)
        ax.set_title(f"{record.index}. {record.title}", fontsize=10, fontproperties=font_prop)

    for ax in axes[len(records):]:
        ax.axis("off")

    fig.suptitle("시별 상위 감정 (KPoEM)", fontsize=13, fontproperties=font_prop)
    fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)