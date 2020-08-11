# 最適化アルゴリズム
## 実装したアルゴリズム
以下のアルゴリズムを実装しています。
- 遺伝的アルゴリズム
  - GA
- 差分進化
  - DE
  - JADE
  - SHADE
- NelderMead

## How to run
test.pyを参照。
以下はDEの場合の例です。
1. 目的関数や次元等を指定してインスタンスを生成
```
solver = DE(func, ndim=10)
```
1. solveメソッドにより最適化を実行
```
solver.solve()
```
1. dispメソッドにより結果を表示
```
solver.disp(disp_x=True)
```
