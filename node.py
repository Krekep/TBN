import math
from typing import Optional


class BayesNode:
    def __init__(self, cpt: dict[str | tuple[str, ...], list[float]], parents: Optional[list]):
        """Bayes network node data structure.

        Parameters
        ----------
        cpt: dict
            Dictionary of conditional probability tables
        parents: list
            list of parents of each node
        """
        for s, p in cpt.items():
            if not math.isclose(sum(p), 1):
                raise ValueError("Sum of probabilities should be equal to 1")

        self.cpt = cpt
        self.parents = parents

    def get_cpt(self) -> dict[str | tuple[str, ...], list[float]]:
        return self.cpt

    def get_parents(self) -> Optional[list]:
        return self.parents
