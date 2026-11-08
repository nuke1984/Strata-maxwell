# iq3_s-planar - answers at temperature 0, first run

## city

```
Firenze, culla del Rinascimento italiano, è una città dove storia e arte si fondono in modo armonioso. Fondata dai Romani come *Florentia*, ha raggiunto il suo apice nel XV secolo sotto la guida della famiglia Medici, diventando un centro nevralgico per intellettuali, artisti e scienziati. Qui hanno lavorato geni assoluti come Leonardo da Vinci, Michelangelo e Botticelli, lasciando un patrimonio artistico inestimabile.

Per visitare Firenze, è imperativo iniziare dalla Cattedrale di Santa Maria del Fiore con la sua iconica cupola brunelleschiana e il Campanile di Giotto. Il Ponte Vecchio, unico ponte medievale sopravvissuto alla guerra, offre scorci suggestivi sull'Arno. Non si può perdere la Galleria degli Uffizi, uno dei musei più importanti al mondo per l'arte rinascimentale, né Piazza della Signoria con il Palazzo Vecchio e le sue statue all'aperto. Passeggiare tra le sue strade significa immergersi in un museo a cielo aperto, dove ogni angolo racconta secoli di bellezza e innovazione culturale.
```

## code

```
```python
from typing import List


def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """
    Merge overlapping or adjacent intervals and return the sorted list of merged intervals.

    Args:
        intervals: A list of intervals, each represented as [start, end].

    Returns:
        A list of non-overlapping, non-adjacent merged intervals, sorted by start time.
    """
    if not intervals:
        return []

    # Sort intervals by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged: List[List[int]] = [sorted_intervals[0][:]]

    for current in sorted_intervals[1:]:
        last = merged[-1]
        # If the current interval overlaps or is adjacent to the last merged interval
        if current[0] <= last[1]:
            # Extend the end of the last merged interval if needed
            last[1] = max(last[1], current[1])
        else:
            merged.append(current[:])

    return merged


# Test cases
if __name__ == "__main__":
    # Test case 1: Overlapping intervals
    intervals1 = [[1, 3], [2, 6], [8, 10], [15, 18]]
    result1 = merge_intervals(intervals1)
    assert result1 == [[1, 6], [8, 10], [15, 18]], f"Test case 1 failed: {result1}"

    # Test case 2: Adjacent intervals and unsorted input
    intervals2 = [[4, 7], [1, 3], [3, 5], [9, 10]]
    result2 = merge_intervals(intervals2)
    assert result2 == [[1, 7], [9, 10]], f"Test case 2 failed: {result2}"

    print("All test cases passed!")
```
```
