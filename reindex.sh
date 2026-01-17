#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

echo "🧹 Cleaning up old RAG Embeddings (Fixing Dimension Mismatch)..."
rm -rf data/chroma_db

echo "🔍 Regenerating MOCs..."
python3 -c "from adapters.obsidian.vault_manager import ObsidianGardener; ObsidianGardener().generate_index()"
echo "📊 Re-indexing RAG Knowledge Base..."
python3 -c "from core.services.rag_service import ObsidianRAG; from config import ProjectConfig; ObsidianRAG().index_vault(ProjectConfig.OBSIDIAN_VAULT)"
echo "✅ All indexes updated!"