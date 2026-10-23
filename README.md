# Electrostatic-bvp

## 项目简介

使用 Python 数值求解二维静电场边值问题：给定区域边界上的电势，计算区域内的电势分布，并由电势梯度得到电场。

该项目聚焦二维均匀介质、无自由电荷区域中的 Laplace 方程，采用有限差分法。研究算例依次为矩形区域解析解验证，以及有限宽平行板电容器的边缘电场和参数变化。

## 项目结构
```text
.
├── configs/
│   └── rectangle.json
├── docs/
│   ├── CHANGELOG.md
│   ├── ROADMAP.md
│   └── VALIDATION.md
├── results/             # 本地生成的数值结果与验收报告，不纳入 Git
├── scripts/
│   ├── format_input.py
│   └── validate_m1.py
├── src/electrostatic_bvp/
│   ├── config.py
│   ├── solver.py
│   └── main.py
├── tests/               # 测试与数值验证
├── pyproject.toml       # Python 项目配置
├── uv.lock
└── README.md
```

## 代码实现

- **离散方法：** 均匀正方网格上的五点有限差分。
- **基础迭代：** Jacobi 迭代，设置停止阈值 `tol` 和最大迭代次数 `Nmax`。
- **解析对照：** 矩形正弦边界算例提供解析解，并记录内部节点的最大绝对误差和 RMSE。
- **电场：** 由电势计算 \(\mathbf{E}=-\nabla\varphi\)，内部用二阶中心差分，边界用二阶单边差分。
- **当前输出：** `solution.npz` 保存网格、电势、解析解、绝对误差与电场分量；`summary.json` 保存配置、网格、收敛状态、离散 Laplace 残差及误差指标。
- **运行依赖：** Python 3.11 及以上、NumPy；测试使用标准库 `unittest`。

详细阶段见[开发路线图](docs/ROADMAP.md)。

## 项目状态

**当前版本：** v0.2.0 — M1 MVP，矩形区域求解与解析验证已完成。

已核验固定边界、Jacobi 更新与停止条件、解析误差、电场方向、网格加密趋势及真实命令行输出。详细方法和复现命令见[数值验收说明](docs/VALIDATION.md)。平行板电容器、绘图、计算域敏感性和 COMSOL 对照仍按路线图推进。

从项目根目录安装锁定依赖并运行默认算例：

```bash
uv sync --locked
PYTHONPATH=src uv run --locked python -m electrostatic_bvp.main \
  --config configs/rectangle.json \
  --output-dir results/rectangle
```

默认算例为 \(L_x=1.5, L_y=1, V_0=1, h=0.05\)，上边界为 \(V_0\sin(\pi x/L_x)\)，两侧和底边为零；解析解为

\[
\varphi(x,y)=V_0\sin(\pi x/L_x)\frac{\sinh(\pi y/L_x)}{\sinh(\pi L_y/L_x)}.
\]

生成其他参数配置时，输入脚本与求解器使用同一套校验规则：

```bash
PYTHONPATH=src uv run --locked python scripts/format_input.py \
  --lx 1.5 --ly 1 --v0 1 --h 0.025 --tol 1e-10 --nmax 100000 \
  --output results/custom-config.json
```

将生成路径传给主程序的 `--config` 即可计算。`h` 必须将两个边长分成至少两个整数区间；`tol` 是相邻迭代间内部电势最大变化量的绝对阈值，不等同于解析误差。

所有二维数组按 `(y, x)` 排列，`x` 和 `y` 是一维坐标。NPZ 包含 `x`、`y`、`potential`、`analytic_potential`、`absolute_error`、`electric_field_x`、`electric_field_y`，可以用 `numpy.load(..., allow_pickle=False)` 读取。长度与电压采用一致单位；若输入为米和伏特，电场为伏特/米。误差指标只统计内部节点，离散残差为内部节点上 \(|\nabla_h^2\varphi|\) 的最大值。

命令退出码：`0` 表示达到停止阈值；`1` 表示达到最大迭代次数，此时仍保存未收敛结果和诊断；`2` 表示配置或文件读写失败。不同算例应使用不同输出目录，同名文件会被覆盖。

完整复核：

```bash
PYTHONPATH=src uv run --locked python -W error -m unittest discover -s tests -v
PYTHONPATH=src uv run --locked python -W error scripts/validate_m1.py
```

数值验收生成 `results/m1-validation/report.json` 和各算例的 NPZ/JSON。验证脚本、配置、测试和说明纳入版本管理，生成的验证结果保留在本地。
