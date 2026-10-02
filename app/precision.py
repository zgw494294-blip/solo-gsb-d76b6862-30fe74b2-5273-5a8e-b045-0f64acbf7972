"""批次内同一样本多次进样的精密度检查（纯函数，全部使用精确有理数运算）。

针对可复核批次中**已确认**的空白扣除结果：逐样本取目标参考峰的校正面积
除以第一枚内标（``is_ref_id_1``）的校正面积作为面积比，对 2～8 个面积比
精确计算均值 μ、样本方差 s²（分母 N−1）与 q = s²/μ²，并以
``10000·q ≤ max_rsd_percent²`` 判定相对标准偏差（RSD，百分数）是否不超过
阈值（等号通过）。零目标面积保留为面积比 0；μ = 0 时 RSD 无定义。

本模块只读取调用方传入的批次快照（来自存储层的单次完整读取），
不写库、不改变确认状态或版本。错误优先级与规格一致：
未知样本 404 → 未确认 409 → 其余非法输入 422。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .assignment import AssignmentError
from .storage import SampleNotConfirmed, SampleNotFound


def _frac_out(value: Fraction) -> dict[str, int]:
    """把约分有理数序列化为 {n, d}，分母恒为正。"""
    return {"n": value.numerator, "d": value.denominator}


def precision_check(
    batch: dict[str, Any],
    sample_ids: list[str],
    target_ref_id: str,
    max_rsd_percent: int,
) -> dict[str, Any]:
    """精密度检查主流程，返回可直接序列化的结果字典。

    ``batch`` 为单次完整读取得到的批次快照；``sample_ids`` 为 2～8 个互异
    样本 id（模型层已保证），面积比按此顺序返回。
    """
    samples_by_id = {s["id"]: s for s in batch["samples"]}

    # —— 未知样本 -> 404（按请求顺序取第一个缺失者）——
    requested = []
    for sid in sample_ids:
        sample = samples_by_id.get(sid)
        if sample is None:
            raise SampleNotFound(
                f"sample {sid!r} not found in batch {batch['batch_id']!r}"
            )
        requested.append(sample)

    # —— 未确认 -> 409（只使用已确认的扣除结果）——
    for sid, sample in zip(sample_ids, requested):
        if not sample["confirmed"]:
            raise SampleNotConfirmed(f"sample {sid!r} is not confirmed")

    # —— 目标参考 id 必须存在于参考峰表且不是内标 ——
    ref_ids = [p["id"] for p in batch["reference_peaks"]]
    if target_ref_id not in ref_ids:
        raise AssignmentError("target_ref_id not found in reference peaks")
    if target_ref_id in (batch["is_ref_id_1"], batch["is_ref_id_2"]):
        raise AssignmentError("target peak must not be an internal standard")

    # —— 逐样本面积比 = 目标校正面积 ÷ 第一内标校正面积（按请求顺序）——
    is1 = batch["is_ref_id_1"]
    ratios: list[tuple[str, Fraction]] = []
    for sid, sample in zip(sample_ids, requested):
        # 归属为一对一，每个参考 id 至多对应一个峰；未归属峰（ref_id 为
        # null）不参与。
        area_by_ref = {
            sp["ref_id"]: Fraction(
                sp["corrected_area"]["n"], sp["corrected_area"]["d"]
            )
            for sp in sample["result"]["sample_peaks"]
            if sp["ref_id"] is not None
        }
        target_area = area_by_ref.get(target_ref_id)
        if target_area is None:
            raise AssignmentError(
                f'sample "{sid}": target peak not assigned in confirmed result'
            )
        # 内标在归属中强制配对，必然已归属；其校正面积可能因扣除截零为 0
        is_area = area_by_ref.get(is1, Fraction(0))
        if is_area == 0:
            raise AssignmentError(
                f'sample "{sid}": internal standard 1 corrected area is zero'
            )
        ratios.append((sid, target_area / is_area))  # 零目标面积保留为 0

    # —— 精确矩：均值 μ、样本方差 s²（分母 N−1）、q = s²/μ² ——
    n = len(ratios)
    mu = sum((r for _, r in ratios), Fraction(0)) / n
    if mu == 0:
        raise AssignmentError("mean area ratio is zero; RSD is undefined")
    s2 = sum(((r - mu) ** 2 for _, r in ratios), Fraction(0)) / (n - 1)
    q = s2 / (mu * mu)

    # —— 10000·q ≤ 阈值² 判为合格（等号通过），全程精确比较 ——
    passed = 10000 * q <= max_rsd_percent * max_rsd_percent

    return {
        "batch_id": batch["batch_id"],
        "version": batch["version"],
        "target_ref_id": target_ref_id,
        "max_rsd_percent": max_rsd_percent,
        "area_ratios": [
            {"sample_id": sid, "area_ratio": _frac_out(r)} for sid, r in ratios
        ],
        "mean": _frac_out(mu),
        "variance": _frac_out(s2),
        "q": _frac_out(q),
        "passed": passed,
    }
