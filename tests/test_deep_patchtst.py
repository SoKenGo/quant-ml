# -*- coding: utf-8 -*-
import os, subprocess, sys

# 用 dummy 数据快速冒烟：
# 仅断言脚本能跑通，并打印出平均指标（不校验数值）

def test_patchtst_dummy_smoke():
    cmd = [
    sys.executable, '-m', 'src.models.deep.patchtst_h1',
    '--symbols', 'NVDA',
    '--years', '2024',
    '--use-dummy',
    '--max-epochs', '2',
    ]
    env = os.environ.copy()
    env['SEED'] = '123'
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert '[PatchTST] NVDA' in p.stdout