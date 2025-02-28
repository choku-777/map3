# プロジェクト開発の指示

## 応答言語について
- GitHub Copilot の提案や応答は、**必ず日本語で対応してください。**

## 開発環境のルール

### Python 利用時の注意事項
- **必ず仮想環境を使用すること。**  
  Python を使用する場合は、依存関係の競合や環境間の不整合を避けるために、必ず仮想環境内で開発を行ってください。

- 仮想環境の作成例:
  ```bash
  # Unix/macOS の場合
  python -m venv venv
  source venv/bin/activate

  # Windows の場合
  python -m venv venv
  venv\Scripts\activate
Windows環境ではコマンドの連結方法が異なるため、コマンドを分けて実行しましょう。