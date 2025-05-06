import torch

from .fnn import FNN
from .nn import NN
from .deeponet_strategy import (
    SingleOutputStrategy,
    IndependentStrategy,
    SplitBothStrategy,
    SplitBranchStrategy,
    SplitTrunkStrategy,
)

class DeepONetCartesianProd(NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net
            should be the same for all strategies except "split_branch" and "split_trunk".
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        num_outputs (integer): Number of outputs. In case of multiple outputs, i.e., `num_outputs` > 1,
            `multi_output_strategy` below should be set.
        multi_output_strategy (str or None): ``None``, "independent", "split_both", "split_branch" or
            "split_trunk". It makes sense to set in case of multiple outputs.

            - None
            Classical implementation of DeepONet with a single output.
            Cannot be used with `num_outputs` > 1.

            - independent
            Use `num_outputs` independent DeepONets, and each DeepONet outputs only
            one function.

            - split_both
            Split the outputs of both the branch net and the trunk net into `num_outputs`
            groups, and then the kth group outputs the kth solution.

            - split_branch
            Split the branch net and share the trunk net. The width of the last layer
            in the branch net should be equal to the one in the trunk net multiplied
            by the number of outputs.

            - split_trunk
            Split the trunk net and share the branch net. The width of the last layer
            in the trunk net should be equal to the one in the branch net multiplied
            by the number of outputs.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        num_outputs=1,
        multi_output_strategy=None,
    ):
        super().__init__()
        self.activation_branch = self.activation_trunk = torch.nn.ReLU()
        self.kernel_initializer = torch.nn.init.xavier_normal
        self.num_outputs = num_outputs
        self.multi_output_strategy = {
            None: SingleOutputStrategy,
            "independent": IndependentStrategy,
            "split_both": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
        }[multi_output_strategy](self)
        self.branch, self.trunk = self.multi_output_strategy.build(layer_sizes_branch, layer_sizes_trunk)
        if isinstance(self.branch, list):
            self.branch = torch.nn.ModuleList(self.branch)
        if isinstance(self.trunk, list):
            self.trunk = torch.nn.ModuleList(self.trunk)
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)]
        )

    def build_branch_net(self, layer_sizes_branch):
        # User-defined network
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        # Fully connected network
        return FNN(layer_sizes_branch, self.activation_branch, self.kernel_initializer)

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(layer_sizes_trunk, self.activation_trunk, self.kernel_initializer)

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = torch.einsum("bi,ni->bn", x_func, x_loc)
        y += self.b[index]
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return torch.stack(ys, dim=2)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x