"""进样精密度检查 HTTP 接口测试（TestClient 走真实 ASGI 栈）。

覆盖：
- 2～8 个互异已确认样本、按请求顺序返回面积比；μ、s²（分母 N−1）、q
  均为约分分数；10000·q ≤ 阈值² 判合格且等号通过；
- 零目标面积保留（μ 非零时正常计算）；
- 未知批次/样本 404、未确认 409、目标未归属/内标面积为零/目标即内标/
  目标参考不存在/μ=0 422（逐样本错误指出样本 id）；
- 请求结构 422：样本数、互异、阈值范围、严格类型、缺字段；
- 只读：版本/确认状态/结果不变，读取只有 BEGIN/SELECT/COMMIT、无写语句；
- 与 GET 相同的“一次完整读取”快照语义：读取暂停期间被确认，放行后
  仍按旧快照返回 409，不拼接新确认状态。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import batches
from app.main import app


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: str):
    db = str(Path(tmp_path) / "batches.db")
    batches.reset_store(db)
    yield db
    batches.reset_store(db)


client = TestClient(app)

# ---------------------------------------------------------------------------
# 批次构造辅助
# ---------------------------------------------------------------------------

REFS2 = [
    {"id": "S1", "time_ms": 3},
    {"id": "S2", "time_ms": 7},
]
REFS3 = [
    {"id": "S1", "time_ms": 1},
    {"id": "T", "time_ms": 3},
    {"id": "S2", "time_ms": 5},
]
# 多一枚永不归属的参考 U（2 个实测峰、3 个参考峰）
REFS_U = [
    {"id": "S1", "time_ms": 3},
    {"id": "S2", "time_ms": 7},
    {"id": "U", "time_ms": 11},
]

TIMES5 = [0, 3, 4, 7, 10]
TIMES7 = list(range(7))


def run(intensities, times=TIMES5, apex=(1, 3), is1=0, is2=1):
    """5 点两峰运行（内标即两枚峰）。"""
    assert len(times) == len(intensities)
    return {
        "samples": [
            {"time_ms": t, "intensity": y} for t, y in zip(times, intensities)
        ],
        "start_index": 0,
        "end_index": len(times) - 1,
        "apex_indices": list(apex),
        "is_apex_idx_1": is1,
        "is_apex_idx_2": is2,
    }


def run3(intensities):
    """7 点三峰运行：apex 0=S1(IS1, t=1)、apex 1=T(目标, t=3)、
    apex 2=S2(IS2, t=5)。"""
    assert len(intensities) == 7
    return run(intensities, times=TIMES7, apex=(1, 3, 5), is1=0, is2=2)


def create_batch(payload):
    r = client.post("/v1/blank-subtraction-batches", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def confirm_all(bid, ids):
    for ord_, sid in enumerate(ids, start=1):
        r = client.post(
            f"/v1/blank-subtraction-batches/{bid}/confirm",
            json={"sample_id": sid, "expected_version": ord_},
        )
        assert r.status_code == 200, r.text


def batch3(
    *,
    samples,
    blank=(1, 1, 1, 3, 1, 1, 1),
    refs=REFS3,
    is1="S1",
    is2="S2",
    tolerance=0,
):
    """samples: [(id, 7 点强度), ...]；blank 为 7 点强度。"""
    return {
        "reference_peaks": refs,
        "is_ref_id_1": is1,
        "is_ref_id_2": is2,
        "tolerance_ms": tolerance,
        "blank": run3(list(blank)),
        "samples": [{"id": sid, **run3(list(y))} for sid, y in samples],
    }


def batch2(*, samples, blank=(1, 1, 1, 2, 1), refs=REFS2):
    return {
        "reference_peaks": refs,
        "is_ref_id_1": "S1",
        "is_ref_id_2": "S2",
        "tolerance_ms": 0,
        "blank": run(list(blank)),
        "samples": [
            {"id": sid, **run(list(y))} for sid, y in samples
        ],
    }


def check(bid, body):
    return client.post(
        f"/v1/blank-subtraction-batches/{bid}/precision-check", json=body
    )


# 主数据：blank 各峰面积 S1=0、T=2、S2=0；样本 S1/S2 面积 4，
# 目标 T 面积随样本 4/6/8 => 校正 2/4/6，面积比 1/2、1、3/2。
MAIN = [
    ("s1", (1, 5, 1, 5, 1, 5, 1)),
    ("s2", (1, 5, 1, 7, 1, 5, 1)),
    ("s3", (1, 5, 1, 9, 1, 5, 1)),
]
IDS = ["s1", "s2", "s3"]


def make_confirmed_main():
    d = create_batch(batch3(samples=MAIN))
    confirm_all(d["batch_id"], IDS)
    return d


# ---------------------------------------------------------------------------
# 正常路径：精确分数、顺序、阈值等号
# ---------------------------------------------------------------------------

def test_precision_check_ratios_mean_variance_q_and_equality_pass():
    d = make_confirmed_main()
    r = check(
        d["batch_id"],
        {"sample_ids": IDS, "target_ref_id": "T", "max_rsd_percent": 50},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 版本取自该次完整读取（三次确认后为 4）；检查本身不改版本
    assert body["batch_id"] == d["batch_id"]
    assert body["version"] == 4
    assert body["target_ref_id"] == "T"
    assert body["max_rsd_percent"] == 50
    # 面积比按请求顺序：T 校正面积 2/4/6，第一内标 S1 校正面积均为 4
    assert body["samples"] == [
        {"sample_id": "s1", "area_ratio": {"n": 1, "d": 2}},
        {"sample_id": "s2", "area_ratio": {"n": 1, "d": 1}},
        {"sample_id": "s3", "area_ratio": {"n": 3, "d": 2}},
    ]
    # μ=1，s²=((−1/2)²+0²+(1/2)²)/2=1/4，q=s²/μ²=1/4（RSD=50%）
    assert body["mean"] == {"n": 1, "d": 1}
    assert body["sample_variance"] == {"n": 1, "d": 4}
    assert body["q"] == {"n": 1, "d": 4}
    # 10000·q = 2500 = 50²：等号通过
    assert body["passed"] is True


def test_threshold_just_below_fails():
    d = make_confirmed_main()
    r = check(
        d["batch_id"],
        {"sample_ids": IDS, "target_ref_id": "T", "max_rsd_percent": 49},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is False
    assert body["q"] == {"n": 1, "d": 4}  # 10000·q=2500 > 2401=49²


def test_zero_threshold_passes_only_for_identical_ratios():
    # 四个进样：a/b/c 完全一致，d 目标面积不同；阈值 0 仅在 s²=0 时通过
    d = create_batch(
        batch3(
            samples=[
                ("a", (1, 5, 1, 7, 1, 5, 1)),
                ("b", (1, 5, 1, 7, 1, 5, 1)),
                ("c", (1, 5, 1, 7, 1, 5, 1)),
                ("d", (1, 5, 1, 9, 1, 5, 1)),
            ]
        )
    )
    confirm_all(d["batch_id"], ["a", "b", "c", "d"])
    r = check(
        d["batch_id"],
        {"sample_ids": ["a", "b", "c"], "target_ref_id": "T",
         "max_rsd_percent": 0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["q"] == {"n": 0, "d": 1}
    assert body["passed"] is True  # 等号（10000·0 = 0²）通过

    # 有离散时阈值 0 不通过（q>0）
    r = check(
        d["batch_id"],
        {"sample_ids": ["a", "d"], "target_ref_id": "T",
         "max_rsd_percent": 0},
    )
    assert r.status_code == 200
    assert r.json()["passed"] is False


def test_results_follow_request_order_not_batch_order():
    d = make_confirmed_main()
    r = check(
        d["batch_id"],
        {"sample_ids": ["s3", "s1", "s2"], "target_ref_id": "T",
         "max_rsd_percent": 100},
    )
    assert r.status_code == 200
    body = r.json()
    assert [s["sample_id"] for s in body["samples"]] == ["s3", "s1", "s2"]
    assert body["samples"][0]["area_ratio"] == {"n": 3, "d": 2}
    assert body["samples"][1]["area_ratio"] == {"n": 1, "d": 2}
    # μ/s²/q 对同一样本集合与顺序无关
    assert body["mean"] == {"n": 1, "d": 1}
    assert body["q"] == {"n": 1, "d": 4}


def test_two_and_eight_samples_bounds():
    for n in (2, 8):
        ids = [f"s{i}" for i in range(n)]
        d = create_batch(
            batch3(
                samples=[(sid, (1, 5, 1, 7, 1, 5, 1)) for sid in ids]
            )
        )
        confirm_all(d["batch_id"], ids)
        r = check(
            d["batch_id"],
            {"sample_ids": ids, "target_ref_id": "T",
             "max_rsd_percent": 0},
        )
        assert r.status_code == 200, r.text
        assert len(r.json()["samples"]) == n


def test_sample_variance_uses_n_minus_one_and_is_reduced():
    # 面积比 1/2、3/2：μ=1，总体方差为 1/4、样本方差（N−1）为 1/2
    d = make_confirmed_main()
    r = check(
        d["batch_id"],
        {"sample_ids": ["s1", "s3"], "target_ref_id": "T",
         "max_rsd_percent": 100},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mean"] == {"n": 1, "d": 1}
    assert body["sample_variance"] == {"n": 1, "d": 2}
    assert body["q"] == {"n": 1, "d": 2}
    # RSD≈70.71%：阈值 71 通过、70 不通过（10000·1/2=5000）
    assert body["passed"] is True
    r2 = check(
        d["batch_id"],
        {"sample_ids": ["s1", "s3"], "target_ref_id": "T",
         "max_rsd_percent": 70},
    )
    assert r2.json()["passed"] is False


def test_zero_target_area_kept_when_mean_nonzero():
    # blank T 峰更强：样本 T 校正面积全部截零；再补一个非零样本
    # s1: T 校正 0（4−6 截零）、S1 校正 4；s2: T 校正 2、S1 校正 4
    d = create_batch(
        batch3(
            blank=(1, 1, 1, 7, 1, 1, 1),
            samples=[
                ("s1", (1, 5, 1, 5, 1, 5, 1)),
                ("s2", (1, 5, 1, 9, 1, 5, 1)),
            ],
        )
    )
    confirm_all(d["batch_id"], ["s1", "s2"])
    r = check(
        d["batch_id"],
        {"sample_ids": ["s1", "s2"], "target_ref_id": "T",
         "max_rsd_percent": 100},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # 零目标面积保留为面积比 0；μ=1/4 非零
    assert body["samples"] == [
        {"sample_id": "s1", "area_ratio": {"n": 0, "d": 1}},
        {"sample_id": "s2", "area_ratio": {"n": 1, "d": 2}},
    ]
    assert body["mean"] == {"n": 1, "d": 4}
    # s² = (1/2·(0−1/4)² + (1/2−1/4)²) = 1/8
    assert body["sample_variance"] == {"n": 1, "d": 8}
    assert body["q"] == {"n": 2, "d": 1}


# ---------------------------------------------------------------------------
# 404：未知批次 / 未知样本
# ---------------------------------------------------------------------------

def test_unknown_batch_404():
    r = check(
        "deadbeefdeadbeefdeadbeefdeadbeef",
        {"sample_ids": ["a", "b"], "target_ref_id": "T",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 404


def test_unknown_sample_404():
    d = make_confirmed_main()
    r = check(
        d["batch_id"],
        {"sample_ids": ["s1", "zzz"], "target_ref_id": "T",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 404
    assert "zzz" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 409：存在未确认样本（先于 422 类业务错误）
# ---------------------------------------------------------------------------

def test_unconfirmed_sample_409_names_sample():
    d = create_batch(batch3(samples=MAIN))  # 全部未确认
    r = check(
        d["batch_id"],
        {"sample_ids": ["s1", "s2"], "target_ref_id": "T",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 409
    assert "s1" in r.json()["detail"]


def test_one_unconfirmed_among_confirmed_409():
    d = create_batch(batch3(samples=MAIN))
    # 只确认 s1、s3（版本前进到 3）
    for sid, ver in (("s1", 1), ("s3", 2)):
        r = client.post(
            f"/v1/blank-subtraction-batches/{d['batch_id']}/confirm",
            json={"sample_id": sid, "expected_version": ver},
        )
        assert r.status_code == 200
    r = check(
        d["batch_id"],
        {"sample_ids": ["s1", "s3"], "target_ref_id": "T",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 200, r.text
    r = check(
        d["batch_id"],
        {"sample_ids": ["s1", "s2", "s3"], "target_ref_id": "T",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 409
    assert "s2" in r.json()["detail"]


def test_unknown_sample_takes_404_over_409():
    d = create_batch(batch3(samples=MAIN))  # 均未确认
    r = check(
        d["batch_id"],
        {"sample_ids": ["s1", "zzz"], "target_ref_id": "T",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 422：目标/内标/均值
# ---------------------------------------------------------------------------

def test_target_equal_internal_standard_422():
    d = make_confirmed_main()
    r = check(
        d["batch_id"],
        {"sample_ids": IDS, "target_ref_id": "S1",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 422
    assert "internal standard" in r.json()["detail"]
    r = check(
        d["batch_id"],
        {"sample_ids": IDS, "target_ref_id": "S2",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 422


def test_target_ref_not_in_batch_422():
    d = make_confirmed_main()
    r = check(
        d["batch_id"],
        {"sample_ids": IDS, "target_ref_id": "nope",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 422
    assert "nope" in r.json()["detail"]


def test_target_unassigned_422_names_sample():
    # 3 个参考峰（S1、S2、U）但每运行只积出 2 个峰：U 永不归属
    d = create_batch(
        batch2(
            refs=REFS_U,
            samples=[
                ("s1", (1, 3, 1, 4, 1)),
                ("s2", (1, 5, 1, 6, 1)),
            ],
        )
    )
    confirm_all(d["batch_id"], ["s1", "s2"])
    r = check(
        d["batch_id"],
        {"sample_ids": ["s1", "s2"], "target_ref_id": "U",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "s1" in detail
    assert "not assigned" in detail


def test_first_internal_standard_zero_area_422_names_sample():
    # blank 的 S1 峰极强：样本 S1 校正面积全部截零（T 校正面积非零）
    d = create_batch(
        batch3(
            blank=(1, 9, 1, 3, 1, 1, 1),
            samples=[
                ("s1", (1, 5, 1, 5, 1, 5, 1)),
                ("s2", (1, 5, 1, 7, 1, 5, 1)),
            ],
        )
    )
    confirm_all(d["batch_id"], ["s1", "s2"])
    r = check(
        d["batch_id"],
        {"sample_ids": ["s1", "s2"], "target_ref_id": "T",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "s1" in detail
    assert "S1" in detail
    assert "zero" in detail


def test_zero_mean_422_when_all_target_ratios_zero():
    # 两个样本 T 校正面积都为零（μ=0），但 S1 校正面积非零（先过内标关）
    d = create_batch(
        batch3(
            blank=(1, 1, 1, 7, 1, 1, 1),
            samples=[
                ("s1", (1, 5, 1, 5, 1, 5, 1)),
                ("s2", (1, 9, 1, 5, 1, 5, 1)),
            ],
        )
    )
    confirm_all(d["batch_id"], ["s1", "s2"])
    r = check(
        d["batch_id"],
        {"sample_ids": ["s1", "s2"], "target_ref_id": "T",
         "max_rsd_percent": 50},
    )
    assert r.status_code == 422
    assert "mean" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 422：请求结构（Pydantic 严格类型）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "body",
    [
        {"sample_ids": ["s1"], "target_ref_id": "T",
         "max_rsd_percent": 50},  # 少于 2
        {"sample_ids": [f"s{i}" for i in range(9)],
         "target_ref_id": "T", "max_rsd_percent": 50},  # 超过 8
        {"sample_ids": ["s1", "s1"], "target_ref_id": "T",
         "max_rsd_percent": 50},  # 互异
        {"sample_ids": ["s1", ""], "target_ref_id": "T",
         "max_rsd_percent": 50},  # 空 id
        {"sample_ids": ["s1", "s2"], "target_ref_id": "T",
         "max_rsd_percent": -1},  # 阈值下界
        {"sample_ids": ["s1", "s2"], "target_ref_id": "T",
         "max_rsd_percent": 101},  # 阈值上界
        {"sample_ids": ["s1", 2], "target_ref_id": "T",
         "max_rsd_percent": 50},  # id 非字符串
        {"sample_ids": ["s1", "s2"], "target_ref_id": 3,
         "max_rsd_percent": 50},  # 目标非字符串
        {"sample_ids": ["s1", "s2"], "target_ref_id": "T",
         "max_rsd_percent": 50.0},  # 阈值浮点
        {"sample_ids": ["s1", "s2"], "target_ref_id": "T",
         "max_rsd_percent": True},  # 布尔不当整数
        {"target_ref_id": "T", "max_rsd_percent": 50},  # 缺 sample_ids
        {"sample_ids": ["s1", "s2"], "max_rsd_percent": 50},  # 缺 target
        {"sample_ids": ["s1", "s2"], "target_ref_id": "T"},  # 缺阈值
    ],
)
def test_malformed_request_422(body):
    d = make_confirmed_main()
    # 超过 8 个样本时需要先造一个 9 样本批次；这里结构错误在模型层就被拒，
    # 与批次无关，直接断言 422 即可。
    r = check(d["batch_id"], body)
    assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# 只读：不改版本/确认/结果，且只有读语句
# ---------------------------------------------------------------------------

def test_check_does_not_mutate_version_or_confirmation():
    d = make_confirmed_main()
    before = client.get(
        f"/v1/blank-subtraction-batches/{d['batch_id']}"
    ).json()
    for threshold in (10, 50, 90):
        r = check(
            d["batch_id"],
            {"sample_ids": IDS, "target_ref_id": "T",
             "max_rsd_percent": threshold},
        )
        assert r.status_code == 200
        assert r.json()["version"] == before["version"]
    after = client.get(
        f"/v1/blank-subtraction-batches/{d['batch_id']}"
    ).json()
    assert after == before
    # 未确认的批次检查（409）同样不留痕
    d2 = create_batch(batch3(samples=MAIN))
    check(
        d2["batch_id"],
        {"sample_ids": ["s1", "s2"], "target_ref_id": "T",
         "max_rsd_percent": 50},
    )
    again = client.get(
        f"/v1/blank-subtraction-batches/{d2['batch_id']}"
    ).json()
    assert again["version"] == 1
    assert all(not s["confirmed"] for s in again["samples"])


def test_check_issues_no_write_sql():
    d = make_confirmed_main()
    store = batches.get_store()
    real_connect = store._connect
    log: list[str] = []

    class _HookedConn:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, parameters=()):
            log.append(sql.strip().split()[0].upper())
            return self._conn.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    @contextmanager
    def patched():
        with real_connect() as conn:
            try:
                yield _HookedConn(conn)
            finally:
                log.append("CLOSE")

    store._connect = patched
    try:
        r = check(
            d["batch_id"],
            {"sample_ids": IDS, "target_ref_id": "T",
             "max_rsd_percent": 50},
        )
        assert r.status_code == 200
    finally:
        store._connect = real_connect
    # 与 GET 相同的单读事务：BEGIN、两条 SELECT、COMMIT、关闭；无写语句
    assert log == ["BEGIN", "SELECT", "SELECT", "COMMIT", "CLOSE"]


# ---------------------------------------------------------------------------
# “一次完整读取”：快照期间被确认仍按旧快照报 409
# ---------------------------------------------------------------------------

def test_check_uses_single_read_snapshot_not_stitching_new_confirms():
    d = create_batch(batch3(samples=MAIN))  # 全部未确认
    bid = d["batch_id"]
    store = batches.get_store()

    row_read = threading.Event()
    resume = threading.Event()
    fired = threading.Event()

    def hook(sql):
        if "params, blank FROM batches" in sql and not fired.is_set():
            fired.set()
            row_read.set()
            assert resume.wait(timeout=10)

    real_connect = store._connect

    @contextmanager
    def patched():
        with real_connect() as conn:
            class _Conn:
                def execute(self_inner, sql, parameters=()):
                    cur = conn.execute(sql, parameters)
                    hook(sql)
                    return cur

                def __getattr__(self_inner, name):
                    return getattr(conn, name)

            yield _Conn()

    store._connect = patched
    outcome: dict = {}

    def do_check():
        try:
            r = check(
                bid,
                {"sample_ids": IDS, "target_ref_id": "T",
                 "max_rsd_percent": 50},
            )
            outcome["status"] = r.status_code
            outcome["json"] = r.json()
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    t = threading.Thread(target=do_check)
    t.start()
    try:
        assert row_read.wait(timeout=10)
        # 读取暂停期间：另一连接把三个样本全部确认（版本 1 -> 4）
        confirm_all(bid, IDS)
    finally:
        resume.set()
        t.join(timeout=10)
        store._connect = real_connect
    assert not t.is_alive()
    assert "error" not in outcome, outcome.get("error")
    # 快照在确认之前建立：检查只用快照中的未确认结果，返回 409，不拼接
    assert outcome["status"] == 409

    # 放行之后的新读取看到版本 4 与全部已确认，检查通过
    fresh = client.get(f"/v1/blank-subtraction-batches/{bid}").json()
    assert fresh["version"] == 4
    assert all(s["confirmed"] for s in fresh["samples"])
    r = check(
        bid,
        {"sample_ids": IDS, "target_ref_id": "T", "max_rsd_percent": 50},
    )
    assert r.status_code == 200
    assert r.json()["version"] == 4
    assert r.json()["passed"] is True
