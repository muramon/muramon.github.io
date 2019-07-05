## RankNet再開発ブログ

RankNetの再開発した際に，気づいた注意すべき点についてまとめる．
尚，RankNetは"From RankNet to LambdaRank to LambdaMART: An Overview"に基づく．

### RankNetについて

RankNetの仕組み

### batch処理について

RankNetは，クエリについてurlのペア$U_i$と$U_j$についてどちらが関連度が高いか比較し，クエリとurlの関連度の関係を学んでいく．

したがって，クエリについてのurlのペア$U_i$と$U_j$について処理する必要がある．ペアを一つずつ処理していくと，計算時間が膨大にかかるため，バッチ処理するべきである．

triu indiceについて

###
