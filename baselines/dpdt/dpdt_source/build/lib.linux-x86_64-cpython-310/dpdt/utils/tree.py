from binarytree import Node, tree
from numpy import uint8, float64, ndarray


### Policy Nodes
class InfoNode(Node):
    def __init__(self, feature: int, threshold: float, left, right):
        super().__init__("x_" + str(feature) + "≤" + str(round(threshold, 3)))
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class ActionNode(Node):
    def __init__(self, classif: int):
        self.classif = classif
        super().__init__("C_" + str(classif))


def extract_tree(policy: dict, root: list, H: int = 0, zeta: int = 0):
    nb_feat = root.shape[0] // 2
    a = policy[tuple(root.tolist() + [H])][zeta]
    if isinstance(a, uint8) or isinstance(a, ndarray):
        return ActionNode(a), 1, 0
    else:
        feat, thresh = a
        left = root.copy()
        left[nb_feat + feat] = thresh
        right = root.copy()
        right[feat] = thresh
        child_l, nodes_l, depth_l = extract_tree(policy, left, H + 1, zeta)
        child_r, nodes_r, depth_r = extract_tree(policy, right, H + 1, zeta)
        return (
            InfoNode(feat, thresh, child_l, child_r),
            nodes_l + nodes_r + 1,
            max(depth_l, depth_r) + 1,
        )