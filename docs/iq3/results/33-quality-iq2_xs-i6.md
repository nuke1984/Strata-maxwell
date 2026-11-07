# iq2_xs - answers at temperature 0, first run

## city

```
Firenze, capitale del Rinascimento italiano, è una città che fonde magistralmente storia, arte e cultura. Fondata dai Romani e poi trasformata dalla famiglia Medici, ha esercitato un'influenza profonda sull'arte europea. Il suo centro storico, patrimonio UNESCO, offre un viaggio nel tempo attraverso monumenti iconici come il Duomo di Santa Maria del Fiore, con la celebre cupola di Brunelleschi, e il Battistero.

Per gli amanti dell'arte, la Galleria degli Uffizi è imperdibile, ospitando capolavori di Botticelli, Leonardo da Vinci e Michelangelo. Non si può ignorare il Ponte Vecchio, unico ponte medievale sopravvissuto alla guerra, né Piazza della Signoria con la sua Loggia dei Lanzi. Passeggiare per le vie strette del centro rivela botteghe artigiane e palazzi nobiliari. Firenze non è solo un museo a cielo aperto, ma un luogo vivo dove la bellezza architettonica si intrecca con una vivace vita sociale, rendendola una meta essenziale per chiunque desideri comprendere l'essenza dell'arte italiana.
```

## code

```
```python
from typing import List


def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """Merge overlapping or adjacent intervals into non-overlapping intervals.

    Given a list of intervals [start, end], merges those that overlap or are
    adjacent (i.e., the end of one interval equals the start of another).
    Returns a sorted list of merged intervals.

    Args:
        intervals: A list of intervals, each represented as [start, end].

    Returns:
        A sorted list of merged, non-overlapping intervals.

    Examples:
        >>> merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])
        [[1, 6], [8, 10], [15, 18]]

        >>> merge_intervals([[1, 4], [4, 5]])
        [[1, 5]]
    """
    if not intervals:
        return []

    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged: List[List[int]] = []
    current_start, current_end = sorted_intervals[0]

    for interval in sorted_intervals[1:]:
        start, end = interval

        # If the next interval overlaps or is adjacent to the current one
        if start <= current_end:
            # Extend the current interval's end if needed
            current_end = max(current_end, end)
        else:
            # No overlap/adjacency: push the current merged interval
            merged.append([current_start, current_end])
            current_start, current_end = start, end

    # Add the last merged interval
    merged.append([current_start, current_end])

    return merged


# Test cases
if __name__ == "__main__":
    # Test 1: Overlapping intervals
    result1 = merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])
    assert result1 == [[1, 6], [8, 10], [15, 18]], f"Test 1 failed: {result1}"

    # Test 2: Adjacent intervals (end of one equals start of another)
    result2 = merge_intervals([[1, 4], [4, 5]])
    assert result2 == [[1, 5]], f"Test 2 failed: {result2}"

    print("All tests passed.")
```
```
