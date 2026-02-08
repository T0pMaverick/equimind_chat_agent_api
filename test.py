from app.services.vector_store import vector_store
from app.config import settings

# Temporarily override threshold
original_threshold = settings.vector_search_score_threshold
settings.vector_search_score_threshold = 0.0  # Get ALL results

results = vector_store.search('Alliance Finance shares issued 2025 September', top_k=5)

print(f'Original threshold: {original_threshold}')
print(f'Total results found: {len(results)}')
print('\nTop 5 Results:')
for i, r in enumerate(results, 1):
    print(f'{i}. Score: {r["score"]:.4f} | Company: {r["metadata"]["company_name"]}')
    print(f'   Preview: {r["content"][:100]}...\n')
