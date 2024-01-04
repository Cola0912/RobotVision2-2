# RobotVision2-2　課題提出用リポジトリ

# 3Dプリント監視・分析プログラム

このリポジトリは、3Dプリンターの印刷中の画像からプリントの失敗を検出するためのプログラムです。失敗を検出した場合は警告文を表示します。現在は写真からの分析のみに対応していますが、将来的には印刷中のプリンターをリアルタイムで監視し、失敗を検知したらプリンターを停止させる機能を追加する予定です。

## 機能

- 写真から3Dプリントの失敗を検出
- 失敗検出時に警告文の表示

## 技術

このプログラムは以下の技術を使用しています：

- **OpenCV (cv2)**: 画像の読み込み、変換、分析のため
- **NumPy**: 画像データの配列操作
- **Matplotlib**: 分析結果の視覚化
- **Scikit-learn**: 線形回帰分析による特徴抽出

## 実行方法

1. 必要なライブラリをインストールします:

    ```bash
    pip install opencv-python numpy matplotlib scikit-learn
    ```

2. スクリプトを実行する前に、画像ファイルのパスが正しいことを確認してください。
3. スクリプトをローカルPython環境で実行します。

### 実行コマンド

プログラムを実行するには、ターミナルまたはコマンドプロンプトで以下のコマンドを入力してください：

```bash
python imgdetector.py

## 依存関係と実行環境

- Python 3.x
- OpenCV, NumPy, Matplotlib, Scikit-learn
- 画像ファイルへのアクセス権



## 現在想定している失敗パターン

- 印刷物が途中で剥がれてしまう
- フィラメント切れによる途中からの吐出不足
- ノズル内部の詰まりによるフィラメント吐出の停止

## ライセンス

このリポジトリはMITライセンスのもとで公開されています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。
