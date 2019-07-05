# RankNet再開発ブログ

RankNetの再開発した際に，気づいた注意すべき点についてまとめます．
尚，RankNetは"From RankNet to LambdaRank to LambdaMART: An Overview"[1]に基づきます．

## RankNetについて

まず，RankNetがどのようなものかここで紹介します．

### RankNet訓練方法

訓練データはクエリによって分割されている．RankNetは入力の特徴ベクトル$x \in \mathscr{R}^{n}$を$f(x)$にマップします．

クエリが与えられた時，urlのペア$U_i$と$U_j$の特徴ベクトル$x_i$, $x_j$はモデルにより，スコア$s_i = f(x_i), s_j = f(x_j)$は計算されます．

$U_{i} \triangleright U_{j}$は$U_i$が$U_j$よりクエリと関連度が高いことを意味します．モデルの2つの出力は，シグモイド関数によって$U_ i$が$U_ j$よりも上位にランク付けされるべきであるという学習済み確率にマッピングされます．
$$
P_{i j} \equiv P\left(U_{i}\triangleright U_{j}\right) \equiv \frac{1}{1+e^{-\sigma\left(s_{i}-s_{j}\right)}}
$$
2つの出力$(s_i, s_j)$はシグモイド関数によって$P_{ij}$にマッピングされる．$P_{ij}$が真の確率分布$\overline{P}_{i j}$に近づくように交差エントロピーコスト関数を適用し学習します．コスト関数は以下の通り，
$$
C=-\overline{P}_{i j} \log P_{i j}-\left(1-\overline{P}_{i j}\right) \log \left(1-P_{i j}\right)
$$
クエリが与えられた時，$S_{i j} \in\{0, \pm 1\}$，$U_{i} \triangleright U_{j}$の場合$1$, $U_{i} \triangleleft U_{j}$の場合$-1$，$U_{i} = U_{j}$の場合は$0$と定義します．そうすると，$\overline{P}_{i j}=\frac{1}{2}\left(1+S_{i j}\right)$となり，コスト関数は，
$$
C=\frac{1}{2}\left(1-S_{i j}\right) \sigma\left(s_{i}-s_{j}\right)+\log \left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right)
$$
となる．このコスト関数は対称的($i$と$j$の符号変えたら不変)であり，$s_{ij} = 1$の時．
$$
C=\log \left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right)
$$
$s_{ij} = -1$の時，
$$
C=\log \left(1+e^{-\sigma\left(s_{j}-s_{i}\right)}\right)
$$
と表されます．また，$s_i = s_j$の時は$C = \log 2$です．したがって，
$$
\frac{\partial C}{\partial s_{i}}=\sigma\left(\frac{1}{2}\left(1-S_{i j}\right)-\frac{1}{1+e^{\sigma\left(s_{i}-s_{j}\right)}}\right)=-\frac{\partial C}{\partial s_{j}}
$$
と表され，重み$w_{k} \in \mathscr{R}$(モデルのパラメータ)は，SGDでコスト関数を最小化することで更新される．
$$
w_{k} \rightarrow w_{k}-\eta \frac{\partial C}{\partial w_{k}}=w_{k}-\eta\left(\frac{\partial C}{\partial s_{i}} \frac{\partial s_{i}}{\partial w_{k}}+\frac{\partial C}{\partial s_{j}} \frac{\partial s_{j}}{\partial w_{k}}\right)
$$
ここで$\eta$は正の学習係数．

---

## バッチ処理について

RankNetは，クエリについてurlのペア$U_i, U_j$について処理を行うが，実際にペア一つずつ計算すると計算量が膨大になってしまいます．計算量の問題を解決するためにバッチ処理を行います．

今回利用したデータセットMQ2007[2]は，一つのクエリについて，約40個の文書があります．40個の文書には文書のから2個の文書ペアを選択する組み合わせは780通り存在します．この780通り全て計算すると計算量が膨大になるので，バッチ処理を行います．

コスト関数を求めるために$s_i - s_j$を求める必要があるので，まず40個の文書について全てのスコア$s$を計算します．計算できたら，$s_i - s_j$の行列を作ります．
$$
\left(\begin{array}{ccccc}{s_1-s_1} & {\dots} & {s_1-s_i} & {\dots} & {s_i-s_{40}} \\ {\vdots} & {\ddots} & {} & {} & {\vdots} \\ {s_i-s_1} & {} & {s_i-s_i} & {} & {s_i-s_{40}} \\ {\vdots} & {} & {} & {\ddots} & {\vdots} \\ {s_{40}-s_1} & {\cdots} & {s_{40}-s_i} & {\cdots} & {s_{40}-s_{40}}\end{array}\right)
$$
この行列は重複を含むための上三角部分を抽出します．

```python
s_batch = self.model(batch_ranking) #スコア計算
pred_diff = s_batch - s_batch.view(1, -1) #s_i - s_j 行列
#対角成分削除
row_inds, col_inds = np.triu_indices(batch_ranking.size()[0], k=1)
si_sj = pred_diff[row_inds, col_inds] #上三角s_i - s_j 行列
```

si_sjを用いてコスト関数を計算します．

## モデルの構築

今回の実装ではPyTorchを用いました．PyTorchでモデルを定義する方法はいくつかあります．

nn.Moduleを継承して，構成要素をinitに定義して順方向をforwardに記載します．モデルに入力を渡せば自動的にforwardを実行してくれます．

最もシンプルな方法は，

```python
class RankNet(nn.Module):
    def __init__(self, input_dim, D_in, H1, H2):
        super(RankNet, self).__init__()
        self.l1 = nn.Linear(input_dim, D_in) #128
        self.l2 = nn.Linear(D_in, H1)        #128, 64
        self.l3 = nn.Linear(H1, H2)          #64, 32
        self.l4 = nn.Linear(H2, 1)           #32, 1
    def forward(self, batch_ranking=None, batch_stds_labels=None, sigma=1.0):
        s_batch = self.l4(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(batch_ranking)))))))
```

です．活性化関数もinitに書くことができます．

次に**Sequential**．必要な処理をひたすらnn.Sequentialに順番に渡していくだけなので簡単です．

```python
class RankNet(nn.Module):
    def __init__(self, input_dim, D_in, H1, H2):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(input_dim, D_in),
                    nn.ReLU(),
                    nn.Linear(D_in, H1),
                    nn.ReLU(),
                    nn.Linear(H1, H2),
                    nn.ReLU(),
                    nn.Linear(H2, 1)
    def forward(self, batch_ranking=None, batch_stds_labels=None, sigma=1.0):
        s_batch = self.model(batch_ranking)
```

となります．

layerを先にリストにまとめておいてnn.Sequentialに渡す方法もあります．

```python
layers = [nn.Linear(input_dim, D_in),
          nn.ReLU(),
          nn.Linear(D_in, H1),
          nn.ReLU(),
          nn.Linear(H1, H2),
          nn.ReLU(),
          nn.Linear(H2, 1)]

class RankNet(nn.Module):
    def __init__(self, layers):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(*layers)
    def forward(self, batch_ranking=None, batch_stds_labels=None, sigma=1.0):
        s_batch = self.model(batch_ranking)
```

この方法を使うとモデルの構造を変えることが容易になります．

## References

[1] Christopher J.C. Burges . (2010) From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft Research Technical Report MSR-TR-2010-82

[2] Tao Qin and Tie-Yan Liu. (2013) Introducing LETOR 4.0 datasets. arXiv:1306.2597. 



[戻る](https://muramon.github.io)