# Electrostatic-bvp

## 项目简介

使用 Python 数值求解二维静电场边值问题：给定区域边界上的电势，计算区域内的电势分布，并由电势梯度得到电场。

该项目聚焦二维均匀介质、无自由电荷区域中的 Laplace 方程，采用有限差分法。研究算例依次为矩形区域解析解验证，以及有限宽平行板电容器的边缘电场和参数变化。

## 项目结构
```text
.
├── configs/
│   ├── rectangle.json
│   └── parallel_plate.json
├── docs/
│   ├── CHANGELOG.md
│   ├── M2-PLAN.md
│   ├── ROADMAP.md
│   └── VALIDATION.md
├── results/             # 本地生成的数值结果与验收报告，不纳入 Git
├── scripts/
│   ├── format_input.py
│   ├── validate_m1.py
│   ├── validate_m2.py
│   ├── plot_m2_grid_comparison.py
│   └── scan_m2.py
├── src/electrostatic_bvp/
│   ├── config.py
│   ├── cases.py
│   ├── fields.py
│   ├── acceleration.py
│   ├── solver.py
│   ├── visualize.py
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
- **矩形输出：** `solution.npz` 保存网格、电势、解析解、绝对误差与电场分量；`summary.json` 保存配置、网格、收敛状态、离散 Laplace 残差及误差指标。
- **平行板当前能力：** 有限长零厚度极板和零电势外框的电势、电场及图像；极板节点的电场无效，收敛变化量与残差仅统计自由节点。
- **运行依赖：** Python 3.11 及以上、NumPy、Matplotlib；测试使用标准库 `unittest`。

详细阶段见[开发路线图](docs/ROADMAP.md)。

## 项目状态

**当前版本：** v0.2.0 之后的开发版：M1 MVP、M2 的 S1–S3 和 S5 参数扫描已完成；S4 与扫描中的网格电场差仍超过建议尺度，M2 整体验收尚未完成。

已核验矩形算例的解析误差与网格加密趋势，以及平行板的内部约束、自由节点残差、电场方向与掩码。五组平行板几何均已进行网格与扩域比较，但网格电场比较尚未达到建议尺度；较长极板的本次扩域比较也超过该尺度。详细方法和结果见[数值验收说明](docs/VALIDATION.md)；COMSOL 对照仍按路线图推进。

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

平行板电势的起步算例与配置生成：

```bash
PYTHONPATH=src uv run --locked python scripts/format_input.py \
  --case parallel_plate --output results/parallel-plate-config.json
PYTHONPATH=src uv run --locked python -m electrostatic_bvp.main \
  --config results/parallel-plate-config.json --output-dir results/parallel_plate
```

默认板长 `L=4`、间距 `d=1`、电压差 `V0=1`，外框为 `[-6,6] × [-4,4]`，`h=0.1`，外框固定为零电势。对应配置也保存在 `configs/parallel_plate.json`。参数 `--plate-length`、`--gap`、`--domain-half-width`、`--domain-half-height` 可按需调整。外框半宽、半高、半板长和半板间距必须是 `h` 的整数倍；两板之间至少四个网格区间，极板距外框至少两个区间。外框是开放空间的截断近似；检查显示该起步外框和网格还不能满足建议比较尺度。

平行板 `solution.npz` 包含网格、电势、电场分量、固定节点与极板掩码、`field_valid_mask` 和 `analysis_valid_mask`；极板上两分量均为 NaN，读取电场时必须使用 `field_valid_mask`。自由节点的差分模板可以以极板节点为端点，但不跨越极板。`analysis_valid_mask` 进一步排除距离极板端点不超过 `0.2d` 的节点；摘要会标记该半径是否至少有两个网格区间，供后续比较参考，并不保证精度。

`summary.json` 保存原配置、实际端点、收敛信息、自由节点残差及相对于 `Eref=|V0|/d` 的中心场、中央区域最大向量偏差和板外点场强。这些比较量不是有限板解析误差。`V0=0` 时归一化指标为 `null` 并注明原因；板外点若不在计算域内也记为 `null`。输出还包括 `potential.png`、`electric_field.png` 和 `midline.png`；矢量箭头统一长度、靠近极板时缩短，颜色表示场强。未收敛运行的图像会标记 `NOT CONVERGED`。使用不同算例时默认输出目录也不同。

所有二维数组按 `(y, x)` 排列，`x` 和 `y` 是一维坐标。矩形 NPZ 包含 `x`、`y`、`potential`、`analytic_potential`、`absolute_error`、`electric_field_x`、`electric_field_y`，可以用 `numpy.load(..., allow_pickle=False)` 读取。长度与电压采用一致单位；若输入为米和伏特，电场为伏特/米。矩形误差指标只统计内部节点，离散残差为内部自由节点上 \(|\nabla_h^2\varphi|\) 的最大值。

命令退出码：`0` 表示达到停止阈值；`1` 表示达到最大迭代次数，此时仍保存未收敛结果和诊断；`2` 表示配置或文件读写失败。不同算例应使用不同输出目录，同名文件会被覆盖。

完整复核：

```bash
PYTHONPATH=src uv run --locked python -W error -m unittest discover -s tests -v
PYTHONPATH=src uv run --locked python -W error scripts/validate_m1.py
```

数值验收生成 `results/m1-validation/report.json` 和各算例的 NPZ/JSON。验证脚本、配置、测试和说明纳入版本管理，生成的验证结果保留在本地。

平行板 S4 数值可靠性检查可单独运行：

```bash
uv sync --locked --extra s4
PYTHONPATH=src uv run --locked --extra s4 python -W error scripts/validate_m2.py \
  --backend numba --output-dir results/m2-validation
```

该命令会运行多个网格和计算域，可能耗时较长。`--backend numba` 只编译相同的 Jacobi 步骤，不启用 fast-math；默认 `numpy` 后端无需额外依赖。每次计算单独保存数组、配置、诊断、依赖版本和源码摘要；若中断，可用相同参数加 `--resume` 续跑，配置或源码不匹配时拒绝复用。`Nmax` 默认为 200 万，收敛仍以每个算例的 `tol` 为准，未收敛或未通过比较尺度会使验收返回非零退出码。检查范围、尺度与当前结果见[数值验收说明](docs/VALIDATION.md)。

平行板 S5 参数扫描可按以下命令运行：

```bash
uv sync --locked --extra s4
PYTHONPATH=src uv run --locked --extra s4 python -W error scripts/scan_m2.py \
  --backend numba --output-dir results/m2-scan
```

扫描固定 `d=1` 改变板长 `L=2、4、8`，再固定 `L=4` 改变间距 `d=0.5、1、2`，共享基准只计算一次。五组几何各自使用 `h/d=0.1、0.05、0.025` 检查网格，并在 `h/d=0.05` 做一次扩域比较；这次检查不代替最细网格扩域和完整 S4 验收。输出 `metrics.csv`、`report.json`、中线曲线、电势及电场比较图，并保留每次运行的配置、数据、收敛状态、残差、环境和源码摘要。

中断后用相同参数追加 `--resume`。可选的 `--seed-s4-runs <S4的runs目录>` 会先验证基准算例的配置、源码、数值环境和文件哈希，再复制匹配结果；不匹配时拒绝复用。任何算例不收敛或网格/扩域比较超出建议尺度时，脚本仍保存结果并返回 `1`；配置或文件错误返回 `2`。比较尺度不是总误差上界。
