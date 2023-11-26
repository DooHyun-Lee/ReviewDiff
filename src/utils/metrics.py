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
    ans = {}
    for topk in topks :
        # check whether the target is in the topk recommendations 
        hit = np.where(recommendations[:, :topk] == targets, 1, 0).sum()
        ans[topk] = hit / len(targets) 
    return ans 



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
    
    for topk in topks :
        dcg = np.where(recommendations[:, :topk] == targets, 1, 0) / np.log2(np.arange(2, topk + 2)).sum()
        ans[topk] = dcg / len(targets) * np.log2(2)
        
    return ans
    