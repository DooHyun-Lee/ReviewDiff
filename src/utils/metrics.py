from typing import List
import numpy as np 

def HR(recommendations, targets, topks: List[int] = [3, 5, 10, 20, 100]) -> float : 
    """
    Compute hit rate
    Args:
        recommendations : List of recommended items [N, k]
        targets : Target item [N, 1]
        k : Number of recommendations
    Returns:
        Hit rate
    """ 
    assert max(topks) <= recommendations.shape[1], "k is larger than the number of recommendations" 
    num_users = recommendations.shape[0]
    num_hits = {k: np.zeros(num_users) for k in topks}
    for k in topks:
        top_k_recommendations = recommendations[:, :k]
        hits = np.sum(top_k_recommendations == np.expand_dims(targets, -1))
        num_hits[k] = hits
    
    hit_rates = {k: num_hits[k] / num_users for k in topks}
    
    return hit_rates



def NDCG(recommendations, targets,  topks: List[int] = [3, 5, 10, 20, 100]) -> float : 
    """
    Compute normalized discounted cumulative gain
    Args:
        recommendations : List of recommended items [N, k]
        targets : Target item [N, 1]
        k : Number of recommendations
    Returns:
        Normalized discounted cumulative gain
    """
    assert max(topks) <= recommendations.shape[1], "k is larger than the number of recommendations" 
    ans = {}
    num_users = recommendations.shape[0]
    for k in topks:
        top_k_recommendations = recommendations[:, :k]
        hits = np.where(top_k_recommendations == np.expand_dims(targets, -1), 1, 0)
        ndcg = np.sum(hits / np.expand_dims(np.log2(np.arange(2, k + 2)),0)) 
        idcg = 1 / np.log2(2) * num_users
        ans[k] = ndcg / idcg
    
    return ans
    