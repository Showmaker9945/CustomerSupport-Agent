#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "CustomerSupport-Agent setup (uv)"
echo "========================================"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv 未安装。"
  echo "安装方式参考: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "1) 同步依赖"
uv sync --all-groups

echo
echo "2) 准备 .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "已创建 .env，请填写你的 LLM_API_KEY 后再继续。"
  exit 1
fi

if grep -q "LLM_API_KEY=your_qwen_api_key_here" .env; then
  echo "请先在 .env 中填写真实的 LLM_API_KEY。"
  exit 1
fi

echo
echo "3) 下载 TextBlob 语料"
uv run python -m textblob.download_corpora >/dev/null 2>&1 || true

echo
echo "4) 创建本地数据目录"
mkdir -p data/knowledge_base data/chroma_db data/user_memory

echo
echo "完成。常用命令："
echo "  启动服务: uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"
echo "  查看文档: http://localhost:8000/docs"
echo "  运行测试: uv run pytest"
