from typing import List, Callable, Any, Optional
from tqdm import tqdm
import torch

def process_in_batches(
    items: List[Any],
    process_fn: Callable,
    batch_size: int = 16,
    show_progress: bool = False,
    desc: Optional[str] = None
) -> List[Any]:
    """Process items in batches.
    
    Args:
        items: List of items to process
        process_fn: Function that processes a batch of items
        batch_size: Size of each batch
        show_progress: Whether to show progress bar
        desc: Description for progress bar
        
    Returns:
        List of processed results
    """
    results = []
    
    # Create progress bar if requested
    if show_progress:
        pbar = tqdm(range(0, len(items), batch_size), desc=desc)
    else:
        pbar = range(0, len(items), batch_size)
    
    for i in pbar:
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch)
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
    
    return results