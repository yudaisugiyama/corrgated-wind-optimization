# corrugated wing optimization

## Quick Start

### 仮想環境構築(uv環境)

```bash
pip instal uv
uv venv
source .venv/bin/activate
uv sync
```

### コルゲート翼形状作成(16x2=32変数)

- x_b, y_b(32変数)を基に、1000次元に拡張してスプライン補完
- 厚み: 定数t
- 参考文献：https://www.cambridge.org/core/services/aop-cambridge-core/content/view/8A2761FBC15473F2CC76ADA9F9366AA9/S0022112025002058a.pdf/wake_transition_and_aerodynamics_of_a_dragonflyinspired_airfoil.pdf

```bash
uv run shape.py
```

実行結果：
![](img/Figure_1.png)


### 深層強化学習モデル