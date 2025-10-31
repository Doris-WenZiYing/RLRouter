#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import random
from pathlib import Path

# 固定節點順序（列/欄）：r1..r19
ROADM_ORDER = [f"r{i}" for i in range(1, 20)]

# 功率定義（對齊你 19rd+26sf 拓樸）
# A100 → 30, H100 → 25, A100x2 → 60
# 每個 ROADM 底下掛的 SF，其 power 列在對應的 list 中
SF_POWER_MAP = {
    "r1":  [30, 60],      # SF1(A100=30),  SF2(A100x2=60)
    "r2":  [25],          # SF3(H100=25)
    "r3":  [30, 25],      # SF4(A100=30),  SF5(H100=25)
    "r4":  [60],          # SF6(A100x2=60)
    "r5":  [30, 25],      # SF7(A100=30),  SF8(H100=25)
    "r6":  [30, 60],      # SF9(A100=30),  SF10(A100x2=60)
    "r7":  [25],          # SF11(H100=25)
    "r8":  [30],          # SF12(A100=30)
    "r9":  [25, 30],      # SF13(H100=25), SF14(A100=30)
    "r10": [60],          # SF15(A100x2=60)
    "r11": [30],          # SF16(A100=30)
    "r12": [25],          # SF17(H100=25)
    "r13": [30, 30],      # SF18(A100=30), SF19(A100=30)
    "r14": [25, 60],      # SF20(H100=25), SF21(A100x2=60)
    "r15": [30],          # SF22(A100=30)
    "r16": [25],          # SF23(H100=25)
    "r17": [30],          # SF24(A100=30)
    "r18": [60],          # SF25(A100x2=60)
    "r19": [25],          # SF26(H100=25)
}

def make_single_tm(src_row, rng):
    """產一筆 19×19 TM；只有 src_row 那列非 0，其它列全 0；對角線為 0。
       值從目標 ROADM 所掛 SF 的 power 隨機挑一個。"""
    N = len(ROADM_ORDER)
    TM = [[0]*N for _ in range(N)]
    for d in range(N):
        if d == src_row:
            TM[src_row][d] = 0
        else:
            dst = ROADM_ORDER[d]
            powers = SF_POWER_MAP.get(dst, [])
            # 若該 ROADM 沒掛 SF（理論上不會發生），就填 0 以防呆
            TM[src_row][d] = rng.choice(powers) if powers else 0
    return TM

def save_tm_csv(tm, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(tm)

def gen_for_one_topology(toponame, out_root, per_dir_samples, seed, round_robin_src):
    """在 out_root/toponame 底下產 per_dir_samples 份 CSV。"""
    rng = random.Random(seed)  # 可重現
    out_dir = Path(out_root) / toponame
    out_dir.mkdir(parents=True, exist_ok=True)

    N = len(ROADM_ORDER)
    for i in range(per_dir_samples):
        src_row = (i % N) if round_robin_src else rng.randrange(N)
        tm = make_single_tm(src_row, rng)
        fname = f"tm_{toponame}_srcRow{src_row}_{i+1:04d}.csv"
        save_tm_csv(tm, out_dir / fname)

        if i == 0:
            print(f"[{toponame}] sample preview → {fname}")
            for r in tm:
                print(r)
            print("-"*40)

    print(f"[{toponame}] Done. Saved {per_dir_samples} CSVs under: {out_dir.resolve()}")

def main(
    per_dir_samples=30,
    out_root="tms_19nodes_enhanced",
    subdirs=("full_mesh", "linear", "ring", "star"),
    seed=2025,
    round_robin_src=False
):
    # 讓四個拓樸用不同隨機序列（但仍可重現）
    for k, topo in enumerate(subdirs):
        gen_for_one_topology(
            toponame=topo,
            out_root=out_root,
            per_dir_samples=per_dir_samples,
            seed=seed + k,             # 每個拓樸 seed 不同
            round_robin_src=round_robin_src
        )

if __name__ == "__main__":
    # 直接跑就會在 tms_19nodes_enhanced/ 下建立 full_mesh/ linear/ ring/ star/
    # 並各自產 30 份 19×19 CSV
    main()
