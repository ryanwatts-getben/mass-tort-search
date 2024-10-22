from utils import load_config

def score_and_rank_results(search_results):
    scored_results = []
    for result in search_results['matches']:
        score = calculate_custom_score(result)
        scored_results.append((result['id'], score, result['metadata']))
    
    # Sort by score in descending order
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results

def calculate_custom_score(result):
    """Calculate custom score based on multiple factors."""
    # Load scoring weights from config
    config = load_config('config/config.yaml')
    weights = config['scoring']
    
    similarity_score = result['score']
    metadata = result['metadata']
    
    # Example calculations (placeholders for actual logic)
    entity_match_score = 1.0 if 'target_entity' in metadata.get('entities', []) else 0.0
    date_relevance_score = 1.0  # Implement actual date relevance logic
    
    # Calculate final score
    final_score = (
        weights['similarity_weight'] * similarity_score +
        weights['entity_match_weight'] * entity_match_score +
        weights['date_relevance_weight'] * date_relevance_score
    )
    return final_score
