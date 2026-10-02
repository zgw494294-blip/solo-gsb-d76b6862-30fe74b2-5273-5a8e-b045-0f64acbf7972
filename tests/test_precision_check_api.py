"""批次内同一样本多次进样精密度检查的 HTTP 接口测试（TestClient 走真实 ASGI 栈）。

测试数据：参考峰 T@2、S1@6、S2@12（内标 S1/S2），空白全零；
样本峰高 (hT, h1, h2) 对应校正面积 (2·hT, 2·h1, 2·h2)，
故目标/第一内标面积比 = hT/h1。
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
    # 每个测试一个全新的临时 SQLite 文件
    db = str(Path(tmp_path) / "batches.db")
    batches.reset_store(db)
    yield db
    batches.reset_store(db)


client = TestClient(app)

REFS = [
    {"id": "T", "time_ms": 2},
    {"id": "S1", "time_ms": 6},
    {"id": "S2", "time_ms": 12},
]

TIMES = [0, 2, 4, 6, 8, 10, 12, 14, 16]


def run(intensities, times=TIMES, apex_indices=(1, 3, 6), is1=1, is2=2, **over):
    body = {
        "samples": [
            {"time_ms": t, "intensity": y} for t, y in zip(times, intensities)
        ],
        "start_index": 0,
        "end_index": len(times) - 1,
        "apex_indices": list(apex_indices),
        "is_apex_idx_1": is1,
        "is_apex_idx_2": is2,
    }
    body.update(over)
    return body


def sample_run(h_target, h_is1=2, h_is2=1):
    # 空白全零 => 校正面积 = 2×峰高：T=2·h_target, S1=2·h_is1, S2=2·h_is2
    return run([0, h_target, 0, h_is1, 0, 0, h_is2, 0, 0])


BLANK = run([0, 0, 0, 0, 0, 0, 0, 0, 0])


def batch_payload(samples):
    return {
        "reference_peaks": REFS,
        "is_ref_id_1": "S1",
        "is_ref_id_2": "S2",
        "tolerance_ms": 0,
        "blank": BLANK,
        "samples": samples,
    }


def create(samples=None):
    if samples is None:
        # 面积比 1/2、1、3/2 -> μ=1，s²=1/4，q=1/4（RSD 恰为 50%）
        samples = [
            {"id": "r1", **sample_run(1)},
            {"id": "r2", **sample_run(2)},
            {"id": "r3", **sample_run(3)},
        ]
    r = client.post("/v1/blank-subtraction-batches", json=batch_payload(samples))
    assert r.status_code == 201, r.text
    return r.json()


def confirm_all(d):
    bid = d["batch_id"]
    version = d["version"]
    for s in d["samples"]:
        r = client.post(
            f"/v1/blank-subtraction-batches/{bid}/confirm",
            json={"sample_id": s["id"], "expected_version": version},
        )
        assert r.status_code == 200, r.text
        version = r.json()["version"]
    return version


def check(bid, sample_ids=("r1", "r2", "r3"), target="T", max_rsd=50):
    return client.post(
        f"/v1/blank-subtraction-batches/{bid}/precision-check",
        json={
            "sample_ids": list(sample_ids),
            "target_ref_id": target,
            "max_rsd_percent": max_rsd,
        },
    )


# ---------------------------------------------------------------------------
# 正常路径：精确分数、请求顺序、阈值判定
# ---------------------------------------------------------------------------

def test_precision_check_exact_fractions_and_state_unchanged():
    d = create()
    bid = d["batch_id"]
    assert confirm_all(d) == 4
    before = client.get(f"/v1/blank-subtraction-batches/{bid}").json()

    r = check(bid)
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["batch_id"] == bid
    assert got["version"] == 4  # 本次读取看到的版本
    assert got["target_ref_id"] == "T"
    assert got["max_rsd_percent"] == 50
    # 面积比按请求顺序：1/2、1、3/2
    assert got["area_ratios"] == [
        {"sample_id": "r1", "area_ratio": {"n": 1, "d": 2}},
        {"sample_id": "r2", "area_ratio": {"n": 1, "d": 1}},
        {"sample_id": "r3", "area_ratio": {"n": 3, "d": 2}},
    ]
    assert got["mean"] == {"n": 1, "d": 1}
    assert got["variance"] == {"n": 1, "d": 4}  # 分母 N−1
    assert got["q"] == {"n": 1, "d": 4}
    # 10000·q = 2500 = 50²：等号通过
    assert got["passed"] is True

    # 检查不写库：版本、确认状态、结果全部不变
    assert client.get(f"/v1/blank-subtraction-batches/{bid}").json() == before


def test_precision_check_area_ratios_follow_request_order():
    d = create()
    bid = d["batch_id"]
    confirm_all(d)
    r = check(bid, sample_ids=("r3", "r1"))
    assert r.status_code == 200, r.text
    got = r.json()
    assert [a["sample_id"] for a in got["area_ratios"]] == ["r3", "r1"]
    assert [a["area_ratio"] for a in got["area_ratios"]] == [
        {"n": 3, "d": 2},
        {"n": 1, "d": 2},
    ]
    # μ=1，s²=((1/2)²+(1/2)²)/1=1/2，q=1/2；10000·q=5000 > 50² -> 不合格
    assert got["mean"] == {"n": 1, "d": 1}
    assert got["variance"] == {"n": 1, "d": 2}
    assert got["q"] == {"n": 1, "d": 2}
    assert got["passed"] is False


def test_precision_check_threshold_boundary_equality_passes():
    d = create()
    bid = d["batch_id"]
    confirm_all(d)
    # 10000·q = 2500：阈值 50 -> 2500 ≤ 2500 通过；49 -> 2401 < 2500 不通过
    assert check(bid, max_rsd=50).json()["passed"] is True
    assert check(bid, max_rsd=49).json()["passed"] is False
    # 阈值 0 仅当 q=0（全部面积比相等）时通过
    d2 = create([
        {"id": "e1", **sample_run(2)},
        {"id": "e2", **sample_run(2)},
    ])
    confirm_all(d2)
    r = check(d2["batch_id"], sample_ids=("e1", "e2"), max_rsd=0)
    assert r.json()["q"] == {"n": 0, "d": 1}
    assert r.json()["passed"] is True


def test_precision_check_zero_target_area_kept():
    # 零目标面积保留：面积比 0 参与计算（不是缺失、不是错误）
    d = create([
        {"id": "z1", **sample_run(0)},
        {"id": "z2", **sample_run(2)},
    ])
    confirm_all(d)
    r = check(d["batch_id"], sample_ids=("z1", "z2"), max_rsd=100)
    assert r.status_code == 200, r.text
    got = r.json()
    assert [a["area_ratio"] for a in got["area_ratios"]] == [
        {"n": 0, "d": 1},
        {"n": 1, "d": 1},
    ]
    # μ=1/2，s²=1/2，q=(1/2)/(1/4)=2；10000·q=20000 > 100² -> 不合格
    assert got["mean"] == {"n": 1, "d": 2}
    assert got["variance"] == {"n": 1, "d": 2}
    assert got["q"] == {"n": 2, "d": 1}
    assert got["passed"] is False


# ---------------------------------------------------------------------------
# 404 / 409
# ---------------------------------------------------------------------------

def test_precision_check_unknown_batch_404():
    r = check("nope")
    assert r.status_code == 404


def test_precision_check_unknown_sample_404():
    d = create()
    bid = d["batch_id"]
    confirm_all(d)
    r = check(bid, sample_ids=("r1", "zz"))
    assert r.status_code == 404
    assert "zz" in r.json()["detail"]


def test_precision_check_unconfirmed_sample_409_and_state_unchanged():
    d = create()
    bid = d["batch_id"]
    # 只确认 r1、r2（版本到 3），r3 未确认
    for sid, v in (("r1", 1), ("r2", 2)):
        r = client.post(
            f"/v1/blank-subtraction-batches/{bid}/confirm",
            json={"sample_id": sid, "expected_version": v},
        )
        assert r.status_code == 200
    r = check(bid)
    assert r.status_code == 409
    assert "r3" in r.json()["detail"]
    # 全部未确认时按请求顺序报第一个
    d2 = create()
    r = check(d2["batch_id"], sample_ids=("r2", "r1"))
    assert r.status_code == 409
    assert "r2" in r.json()["detail"]
    # 409 不改变确认状态或版本
    d3 = client.get(f"/v1/blank-subtraction-batches/{bid}").json()
    assert d3["version"] == 3
    assert [s["confirmed"] for s in d3["samples"]] == [True, True, False]


def test_precision_check_error_precedence_404_then_409_then_422():
    d = create()
    bid = d["batch_id"]
    # 未知样本 + 非法目标（内标）同时存在：先报 404
    r = check(bid, sample_ids=("r1", "zz"), target="S1")
    assert r.status_code == 404
    # 未确认样本 + 非法目标（内标）同时存在：先报 409
    r = check(bid, sample_ids=("r1", "r2"), target="S1")
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# 422：请求结构、目标 id、逐样本数据问题
# ---------------------------------------------------------------------------

def test_precision_check_request_shape_422():
    d = create()
    bid = d["batch_id"]
    confirm_all(d)
    url = f"/v1/blank-subtraction-batches/{bid}/precision-check"

    def post(**over):
        body = {"sample_ids": ["r1", "r2"], "target_ref_id": "T",
                "max_rsd_percent": 50}
        body.update(over)
        return client.post(url, json=body)

    assert post(sample_ids=["r1", "r1"]).status_code == 422  # 重复 id
    assert post(sample_ids=["r1"]).status_code == 422  # 少于 2 个
    assert post(sample_ids=[f"s{i}" for i in range(9)]).status_code == 422  # 多于 8 个
    assert post(sample_ids=["r1", ""]).status_code == 422  # 空 id
    assert post(sample_ids=["r1", 2]).status_code == 422  # 非字符串 id
    assert post(max_rsd_percent=-1).status_code == 422
    assert post(max_rsd_percent=101).status_code == 422
    assert post(max_rsd_percent=2.5).status_code == 422  # 浮点不截断
    assert post(max_rsd_percent=True).status_code == 422  # 布尔不是整数
    r = client.post(url, json={"sample_ids": ["r1", "r2"]})
    assert r.status_code == 422  # 缺字段


def test_precision_check_target_must_not_be_internal_standard():
    d = create()
    bid = d["batch_id"]
    confirm_all(d)
    for target in ("S1", "S2"):
        r = check(bid, target=target)
        assert r.status_code == 422
        assert "internal standard" in r.json()["detail"]


def test_precision_check_target_unknown_ref_422():
    d = create()
    bid = d["batch_id"]
    confirm_all(d)
    r = check(bid, target="ZZ")
    assert r.status_code == 422
    assert "not found" in r.json()["detail"]


def test_precision_check_target_not_assigned_422_names_sample():
    # rX 的目标峰在 t=3（参考 T 在 t=2，容差 0）-> 目标未归属
    shifted = run([0, 1, 0, 2, 0, 0, 1, 0, 0],
                  times=[0, 3, 4, 6, 8, 10, 12, 14, 16])
    d = create([
        {"id": "ok", **sample_run(1)},
        {"id": "rX", **shifted},
    ])
    confirm_all(d)
    r = check(d["batch_id"], sample_ids=("ok", "rX"))
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert 'sample "rX"' in detail
    assert "not assigned" in detail


def test_precision_check_zero_internal_standard_area_422_names_sample():
    # rZ 的第一内标峰高为 0 -> 校正面积 0，无法归一化
    d = create([
        {"id": "ok", **sample_run(1)},
        {"id": "rZ", **sample_run(1, h_is1=0)},
    ])
    confirm_all(d)
    r = check(d["batch_id"], sample_ids=("ok", "rZ"))
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert 'sample "rZ"' in detail
    assert "internal standard 1" in detail


def test_precision_check_zero_mean_422():
    # 全部目标面积为 0 -> 面积比全 0 -> μ=0，RSD 无定义
    d = create([
        {"id": "z1", **sample_run(0)},
        {"id": "z2", **sample_run(0)},
    ])
    confirm_all(d)
    r = check(d["batch_id"], sample_ids=("z1", "z2"))
    assert r.status_code == 422
    assert "mean" in r.json()["detail"]


# ---------------------------------------------------------------------------
# 只读语义：不写库；单次完整读取不拼接并发提交
# ---------------------------------------------------------------------------

# 批次行 SELECT 的识别片段（区别于版本校验的 SELECT version FROM batches）
BATCH_ROW_SQL = "params, blank FROM batches"


class _HookedConn:
    """委托 sqlite3.Connection：execute 后触发钩子，并记录语句首词。"""

    def __init__(self, conn, hook=None, log=None):
        self._conn = conn
        self._hook = hook
        self._log = log

    def execute(self, sql, parameters=()):
        cur = self._conn.execute(sql, parameters)
        if self._log is not None:
            self._log.append(sql.strip().split()[0].upper())
        if self._hook is not None:
            self._hook(sql)
        return cur

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _patch_connect(store, hook=None, log=None):
    """把 store 的连接换成包装版（仅本实例生效），返回还原函数。"""
    real_connect = store._connect

    @contextmanager
    def patched():
        with real_connect() as conn:
            try:
                yield _HookedConn(conn, hook=hook, log=log)
            finally:
                if log is not None:
                    log.append("CLOSE")

    store._connect = patched
    return lambda: setattr(store, "_connect", real_connect)


def test_precision_check_is_read_only_single_transaction():
    d = create()
    bid = d["batch_id"]
    confirm_all(d)
    log: list[str] = []
    _patch_connect(batches.get_store(), log=log)

    r = check(bid)
    assert r.status_code == 200, r.text
    # 与 GET 相同的只读事务：BEGIN + 两条 SELECT + COMMIT，无任何写语句
    assert log == ["BEGIN", "SELECT", "SELECT", "COMMIT", "CLOSE"]
    assert client.get(
        f"/v1/blank-subtraction-batches/{bid}"
    ).json()["version"] == 4


def test_precision_check_reads_single_complete_snapshot():
    """检查读取暂停期间另一连接合法提交确认：结果只来自完整旧状态。

    复现方式同读取一致性回归测试：检查在批次行 SELECT 后暂停，主线程
    以 expected_version 合法提交对 r2 的确认（版本 2 -> 3），再放行。
    读事务持有旧快照：r2 在快照中未确认 -> 409；绝不会拼出
    “新版本 + 旧确认状态”的混合结果。
    """
    d = create([
        {"id": "r1", **sample_run(1)},
        {"id": "r2", **sample_run(2)},
    ])
    bid = d["batch_id"]
    r = client.post(
        f"/v1/blank-subtraction-batches/{bid}/confirm",
        json={"sample_id": "r1", "expected_version": 1},
    )
    assert r.status_code == 200

    store = batches.get_store()
    row_read = threading.Event()
    resume = threading.Event()
    fired = threading.Event()

    def hook(sql):
        if BATCH_ROW_SQL in sql and not fired.is_set():
            fired.set()
            row_read.set()
            assert resume.wait(timeout=10), "检查线程等待放行超时"

    _patch_connect(store, hook=hook)

    outcome: dict = {}

    def reader():
        try:
            resp = check(bid, sample_ids=("r1", "r2"))
            outcome["status"] = resp.status_code
            outcome["json"] = resp.json()
        except BaseException as exc:
            outcome["error"] = exc

    t = threading.Thread(target=reader)
    t.start()
    try:
        assert row_read.wait(timeout=10), "未等到检查线程进入读事务"
        # 另一连接在检查暂停期间合法提交对 r2 的确认（版本 2 -> 3）
        r = client.post(
            f"/v1/blank-subtraction-batches/{bid}/confirm",
            json={"sample_id": "r2", "expected_version": 2},
        )
        assert r.status_code == 200, r.text
    finally:
        resume.set()
        t.join(timeout=10)
    assert not t.is_alive(), "检查线程未结束"
    assert "error" not in outcome, outcome.get("error")

    # 完整旧状态：r2 未确认 -> 409；且并发确认已正常落库
    assert outcome["status"] == 409
    assert "r2" in outcome["json"]["detail"]
    d2 = client.get(f"/v1/blank-subtraction-batches/{bid}").json()
    assert d2["version"] == 3
    assert [s["confirmed"] for s in d2["samples"]] == [True, True]

    # 提交之后的检查看到完整新状态：版本 3，两个面积比
    r = check(bid, sample_ids=("r1", "r2"))
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["version"] == 3
    assert [a["area_ratio"] for a in got["area_ratios"]] == [
        {"n": 1, "d": 2},
        {"n": 1, "d": 1},
    ]
