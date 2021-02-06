# 13. 運動制御

到達運動学習の計算論的モデルとして

1. フィードバック誤差学習 (feedback error learning; FEL) モデル：目標軌道を必要とする （躍度最小化，トルク最小化などの規範による軌道の計画を事前に要する）．
2. 最適フィードバック制御 (Optimal feedback control; OFC) モデル：目標軌道を必要としない．Kalman filterによる状態推定と推定された状態に基づいて運動指令を生成．

がある．

```{tableofcontents}

```