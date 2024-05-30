import argparse
import random
import re
import numpy as np
import pandas as pd
import torch


def add_common_arg(parser):
    def torch_device(arg):
        if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
            raise argparse.ArgumentTypeError(
                'Wrong device format: {}'.format(arg)
            )

        if arg != 'cpu':
            splited_device = arg.split(':')

            if (not torch.cuda.is_available()) or \
                    (len(splited_device) > 1 and
                     int(splited_device[1]) > torch.cuda.device_count()):
                raise argparse.ArgumentTypeError(
                    'Wrong device: {} is not available'.format(arg)
                )

        return arg

    # Base
    parser.add_argument('--device',
                        type=torch_device, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='Seed')

    return parser


def add_train_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--train_load',
                            type=str,
                            help='Input data in csv format to train')
    common_arg.add_argument('--val_load', type=str,
                            help="Input data in csv format to validation")
    common_arg.add_argument('--model_save',
                            type=str, required=True, default='model.pt',
                            help='Where to save the model')
    common_arg.add_argument('--save_frequency',
                            type=int, default=20,
                            help='How often to save the model')
    common_arg.add_argument('--log_file',
                            type=str, required=False,
                            help='Where to save the log')
    common_arg.add_argument('--config_save',
                            type=str, required=True,
                            help='Where to save the config')
    common_arg.add_argument('--vocab_save',
                            type=str,
                            help='Where to save the vocab')
    common_arg.add_argument('--vocab_load',
                            type=str,
                            help='Where to load the vocab; '
                                 'otherwise it will be evaluated')

    return parser


def add_sample_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load',
                            type=str, required=True,
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, required=True,
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, required=True,
                            help='Where to load the vocab')
    common_arg.add_argument('--n_samples',
                            type=int, required=True,
                            help='Number of samples to sample')
    common_arg.add_argument('--gen_save',
                            type=str, required=True,
                            help='Where to save the gen molecules')
    common_arg.add_argument("--n_batch",
                            type=int, default=32,
                            help="Size of batch")
    common_arg.add_argument("--max_len",
                            type=int, default=100,
                            help="Max of length of SMILES")

    return parser

def add_opt_args(parser):
    # Common
    common_arg = parser.add_argument_group('Common')
    add_common_arg(common_arg)
    common_arg.add_argument('--model_load',
                            type=str, required=True,
                            help='Where to load the model')
    common_arg.add_argument('--config_load',
                            type=str, required=True,
                            help='Where to load the config')
    common_arg.add_argument('--vocab_load',
                            type=str, required=True,
                            help='Where to load the vocab')
    #common_arg.add_argument('--n_samples',
                            type=int, required=True,
                            help='Number of samples to sample')
    common_arg.add_argument('--opt_save',
                            type=str, required=True,
                            help='Where to save the gen molecules')
    #common_arg.add_argument("--n_batch",
    #                        type=int, default=32,
    #                        help="Size of batch")
    common_arg.add_argument("--max_len",
                            type=int, default=100,
                            help="Max of length of SMILES")

    return parser


def read_data_csv(path, config):
    
    if hasattr(config, 'reg_prop_tasks'):
        df = pd.read_csv(path)
        cols = ['logP', 'qed', 'SAS']
        cols.insert(0, 'SELFIES' if config.use_selfies else 'SMILES')
        data = df[cols].values
        return data
    
    return read_smiles_csv(path) if config.use_selfies else read_selfies_csv(path)
    
def read_smiles_csv(path):
    
    df = pd.read_csv(path, usecols=['SMILES'])
    smiles_list = df['SMILES'].astype(str).tolist()
    
    return smiles_list

def read_selfies_csv(path):
        
    df = pd.read_csv(path, usecols=['SELFIES'])
    selfies_list = df['SELFIES'].astype(str).tolist()
    
    return selfies_list


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ManualAdamOpt:
    def __init__(self, f, clf, learning_rate=0.01, max_iter=1000, tolerance=1e-9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimizer with the given parameters.
        
        Parameters:
        f (function): The function to maximize.
        learning_rate (float): Learning rate for the optimizer.
        max_iter (int): Maximum number of iterations.
        tolerance (float): Tolerance for stopping criteria.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): A small constant for numerical stability.
        """
        self.f = f
        self.clf =clf
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

    def numerical_gradient(self, x):
        """
        Compute the numerical gradient of the function at x.
        
        Parameters:
        x (numpy.ndarray): The point at which to compute the gradient.
        
        Returns:
        numpy.ndarray: The gradient of the function at x.
        """
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += self.epsilon
            x_minus[i] -= self.epsilon
            grad[i] = (self.f(x_plus, self.clf) - self.f(x_minus, self.clf)) / (2 * self.epsilon)
        return grad

    def optimize(self, initial_x):
        """
        Perform Adam gradient ascent to find the maximum of the function.
        
        Parameters:
        initial_x (numpy.ndarray): The starting point for the optimization.
        
        Returns:
        numpy.ndarray: The point at which the function is maximized.
        """
        x = initial_x
        m = np.zeros_like(x)  # Initialize the first moment vector
        v = np.zeros_like(x)  # Initialize the second moment vector
        t = 0  # Initialize the time step
        for i in range(self.max_iter):
            t += 1
            self.iterations += 1
            grad = self.numerical_gradient(x)
            
            m = self.beta1 * m + (1 - self.beta1) * grad  # Update biased first moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)  # Update biased second moment estimate

            m_hat = m / (1 - self.beta1 ** t)  # Compute bias-corrected first moment estimate
            v_hat = v / (1 - self.beta2 ** t)  # Compute bias-corrected second moment estimate

            new_x = x + self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)  # Update the parameters
            
            # Check convergence
            if np.linalg.norm(new_x - x) < self.tolerance:
                break
            x = new_x
            print(f"Iteration {i+1}: x = {x}, f(x) = {self.f(x, self.clf)}")
        
        return x
    
    def get_function(self):
        return self.f
    
    def get_gpr(self):
         return self.clf
    
    def get_learning_rate(self):
        return self.learning_rate

    def get_max_iter(self):
        return self.max_iter

    def get_tolerance(self):
        return self.tolerance

    def get_beta1(self):
        return self.beta1

    def get_beta2(self):
        return self.beta2

    def get_epsilon(self):
        return self.epsilon

    def get_iterations(self):
        return self.iterations