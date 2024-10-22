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
    weights = config.get('scoring', {})
    
    similarity_score = result['score']
    metadata = result['metadata']
    
    # Calculate entity match score
    entity_match_score = calculate_entity_match_score(metadata)
    
    # Calculate date relevance score (if applicable)
    date_relevance_score = calculate_date_relevance_score(metadata)
    
    # Calculate final score
    final_score = (
        weights.get('similarity_weight', 0.5) * similarity_score +
        weights.get('entity_match_weight', 0.3) * entity_match_score +
        weights.get('date_relevance_weight', 0.2) * date_relevance_score
    )
    return final_score

def calculate_entity_match_score(metadata):
    """Calculate a score based on the presence and relevance of entities."""
    score = 0
    if 'PROBLEM' in metadata:
        score += len(metadata['PROBLEM']) * 0.2
    if 'TEST' in metadata:
        score += len(metadata['TEST']) * 0.15
    if 'TREATMENT' in metadata:
        score += len(metadata['TREATMENT']) * 0.15
    if 'DRUG' in metadata:
        score += len(metadata['DRUG']) * 0.1
    if 'ANATOMY' in metadata:
        score += len(metadata['ANATOMY']) * 0.1
    return min(score, 1.0)  # Normalize to max of 1.0

def calculate_date_relevance_score(metadata):
    """Calculate a score based on the date relevance of the document."""
    # Implement date relevance logic here
    # For now, return a placeholder score
    return 1.0
