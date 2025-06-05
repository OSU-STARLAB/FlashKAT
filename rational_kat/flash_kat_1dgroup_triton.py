
import torch
import torch.nn as nn
from .kernels.flash_rational_triton import FlashRationalTriton1DGroup
from .kat_1dgroup_torch import Rational_CUDA_A_1DGroup
import json
import os

class FlashKAT_Group(nn.Module):
    def __init__(self, in_features, num_groups, den_groups, m, n, mode, device="cuda"):
        """
        Initialize the KAT_Group module.

        Args:
            num_groups (int): Number of groups for separate processing of input.
            mode (str): Initialization mode, determines weights preset from JSON file.
            device (str): Device to run the module on ('cuda' or 'cpu').
        """
        super(FlashKAT_Group, self).__init__()
        assert device in ["cuda", "cpu"], "Device must be either 'cuda' or 'cpu'."
        #Indexing
        self.m = m
        self.n = n
        self.in_features = in_features
        self.num_groups = num_groups if num_groups > 0 else in_features
        self.den_groups = den_groups if den_groups > 0 else in_features

        self.numerator_v = torch.nn.Parameter(
            torch.zeros(self.num_groups, m)
        )

        if n > 0:
            self.denominator_v = torch.nn.Parameter(
                torch.zeros(self.den_groups, n)
            )

        if m == 6 and n == 4:
            self.rational = FlashRationalTriton1DGroup.apply
            self.initialize(mode=mode)
        else:
            raise NotImplementedError
    
    def initialize(self, mode="gelu"):
        """
        Initialize weights from a JSON file based on the specified mode.

        Args:
            mode (str): The initialization mode to use.
        """
        cfd = os.path.dirname(os.path.realpath(__file__))
        try:
            with open(f'{cfd}/init.json') as json_file:
                data = json.load(json_file)

            # Extract weights from the JSON data
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"])
            weight_numerator = torch.stack([weight_numerator] * self.num_groups, dim=0)
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
            weight_denominator = torch.stack([weight_denominator] * self.den_groups, dim=0)
             
            # Register weights as trainable parameters
            self.numerator_v = nn.Parameter(weight_numerator.float(), requires_grad=True)
            self.denominator_v = nn.Parameter(weight_denominator.float(), requires_grad=True) 

        except FileNotFoundError:
            print("Initialization JSON file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON.")

    def forward(self, input):
        """
        Forward pass of the module.

        Args:
            input (Tensor): 3D or 2D input tensor.

        Returns:
            Tensor: Processed tensor after applying rational function.
        """
        assert input.dim() == 3, "Input tensor must be 3D (batch, length, channels)."
    
    
        # Repeat the weights for all groups
        if self.num_groups == 1:
            numerator_v = self.numerator_v.repeat(self.den_groups, 1)
        else:
            numerator_v = self.numerator_v

        if self.den_groups == 1:
            denominator_v = self.denominator_v.repeat(self.num_groups, 1)
        else:
            denominator_v = self.denominator_v

        x = self.rational(input, numerator_v, denominator_v)

        return x
        
    def extra_repr(self):
        """
        Extra representation of the module for debugging.

        Returns:
            str: String representation of the module's configuration.
        """
        return f'num_groups_num={self.num_groups}, num_groups_den={self.den_groups}, order={self.m}, {self.n}'
    