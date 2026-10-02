"""同一样本多次进样的精密度检查（只读，全部使用精确有理数运算）。

输入是批次的**一次完整读取**（``BatchStore.get`` 的返回：版本、参考峰、
内标、各样本结果与确认状态来自同一 WAL 快照）。仅使用其中**已确认**
样本的扣除结果，逐样本以目标参考峰的约分校正面积除以**第一枚内标**的
约分校正面积，得到面积比；再精确计算均值 μ、样本方差 s²（分母 N−1）
与 q = s²/μ²，以 10000·q ≤ 阈值²（等号通过）判定合格。

检查只读：不写库、不改变确认状态、不改变版本；所有结果直接取自读取到
的快照，不重算扣除，也不会发起第二次读取。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .storage import SampleNotFound


class UnconfirmedResult(Exception):
    """请求的样本在本次读取的快照中未确认，API 层映射为 409。"""


class PrecisionCheckError(Exception):
    """目标/内标未归属、内标面积为零或均值为零等，API 层映射为 422。"""


def _frac_out(value: Fraction) -> dict[str, int]:
    """把约分有理数序列化为 {n, d}，分母恒正。"""
    return {"n": value.numerator, "d": value.denominator}


def _corrected_area_by_ref(
    result: dict[str, Any], ref_id: Any
) -> Fraction | None:
    """取已确认结果中归属给 ref_id 的峰的约分校正面积；未归属返回 None。"""
    for peak in result["sample_peaks"]:
        if peak["ref_id"] == ref_id:
            area = peak["corrected_area"]
            return Fraction(area["n"], area["d"])
    return None


def precision_check(
    batch: dict[str, Any],
    *,
    sample_ids: list[str],
    target_ref_id: str,
    max_rsd_percent: int,
) -> dict[str, Any]:
    """对一次完整批次读取做精密度检查，返回可直接序列化的结果字典。

    sample_ids 的结构约束（2～8 项、互异）由模型层保证；本函数负责：
    未知样本 404（``SampleNotFound``）、未确认 409（``UnconfirmedResult``）、
    目标未归属/内标面积为零/目标参考非法/均值为零 422（``PrecisionCheckError``）。
    """
    by_id = {s["id"]: s for s in batch["samples"]}

    # 先判定全部样本存在，再判定全部已确认：
    # 未知批次/样本一律 404，未确认一律 409，互不掩盖。
    for sid in sample_ids:
        if sid not in by_id:
            raise SampleNotFound(
                f"sample {sid!r} not found in batch {batch['batch_id']!r}"
            )
    for sid in sample_ids:
        if not by_id[sid]["confirmed"]:
            raise UnconfirmedResult(
                f"sample {sid!r} is not confirmed in batch "
                f"{batch['batch_id']!r}"
            )

    # 目标必须是批次参考峰之一，且不得是内标（与具体样本无关的全局校验，
    # 先于逐样本取数）。
    ref_ids = {p["id"] for p in batch["reference_peaks"]}
    if target_ref_id not in ref_ids:
        raise PrecisionCheckError(
            f"target ref id {target_ref_id!r} not found in batch "
            f"{batch['batch_id']!r} reference peaks"
        )
    if target_ref_id in (batch["is_ref_id_1"], batch["is_ref_id_2"]):
        raise PrecisionCheckError(
            f"target ref id {target_ref_id!r} is an internal standard of batch "
            f"{batch['batch_id']!r}"
        )

    # 逐样本取数：目标未归属或第一枚内标面积为零时报 422 并指出样本 id；
    # 目标校正面积为零是合法观测，保留为面积比 0。
    ratios: list[Fraction] = []
    for sid in sample_ids:
        result = by_id[sid]["result"]
        target_area = _corrected_area_by_ref(result, target_ref_id)
        if target_area is None:
            raise PrecisionCheckError(
                f'sample "{sid}": target ref {target_ref_id!r} is not assigned '
                f"in the confirmed result"
            )
        is_area = _corrected_area_by_ref(result, batch["is_ref_id_1"])
        if is_area is None:
            # 创建时内标已强制配对，理论上不会缺失；保留为防御性校验。
            raise PrecisionCheckError(
                f'sample "{sid}": first internal standard '
                f"{batch['is_ref_id_1']!r} is not assigned in the confirmed result"
            )
        if is_area == 0:
            raise PrecisionCheckError(
                f'sample "{sid}": corrected area of the first internal standard '
                f"{batch['is_ref_id_1']!r} is zero"
            )
        ratios.append(target_area / is_area)

    n = len(ratios)  # 2～8（模型层保证），N−1 分母总有定义
    mean = sum(ratios, Fraction(0)) / n
    if mean == 0:
        # 全部面积比为零（目标面积全为零）：q 无定义。
        raise PrecisionCheckError(
            "mean target-to-internal-standard area ratio is zero"
        )
    variance = sum((r - mean) ** 2 for r in ratios) / (n - 1)
    q = variance / (mean ** 2)

    # 10000·q ≤ 阈值² 即 RSD% = 100·s/μ 不超过阈值；全程整数比较，等号通过。
    threshold = Fraction(max_rsd_percent)
    passed = 10000 * q <= threshold ** 2

    return {
        "batch_id": batch["batch_id"],
        "version": batch["version"],
        "target_ref_id": target_ref_id,
        "max_rsd_percent": max_rsd_percent,
        "samples": [
            {"sample_id": sid, "area_ratio": _frac_out(ratio)}
            for sid, ratio in zip(sample_ids, ratios)
        ],
        "mean": _frac_out(mean),
        "sample_variance": _frac_out(variance),
        "q": _frac_out(q),
        "passed": passed,
    }
