# Copyright (c) Facebook, Inc. and its affiliates.

# !/usr/bin/env python3

from abc import abstractmethod
import cvxpy as cp
import itertools
import numpy as np
import numpy.linalg
import torch
import scs
import scipy
from sklearn.datasets import make_low_rank_matrix

class ScsInstance:
    """ Class that transforms a problem in SCS format output by cvxpy
        (i.e. a ParamConeProg) into an object whose components are
        directly usable by SCS
    """
    id_iter = itertools.count()

    def __init__(self, input_problem, cones=None):
        data = {
            'A': input_problem['A'],
            'b': input_problem['b'],
            'c': input_problem['c'],
        }
        self.data = data

        if 'dims' not in input_problem:
            assert cones is not None, ("Either the problem must contain cones"
                                       "or the cones must be provided")
        else:
            cone_dims = input_problem['dims']
            cones = {
                "f": cone_dims.zero,
                "l": cone_dims.nonneg,  # needed to swap from nonpos to nonneg
                "q": cone_dims.soc,
                "ep": cone_dims.exp,
                "s": cone_dims.psd,
            }
        self.cones = cones
        self.instance_id = next(self.id_iter)

    def get_sizes(self):
        """Get the A matrix sizes for SCS problem"""
        sizes = self.data['A'].shape
        return sizes

    @staticmethod
    def create_scaled_instance(A, b, c, D, E, sigma, rho, cones):
        """
           Create new SCS instance with the provided A, b, c, and cones.
           Assumes that A, b, and c are already normalized as needed.
           Track D, E, sigma, rho to allow for rescaling the solution.
        """
        prob = {}
        prob["A"], prob["b"], prob["c"] = A, b, c
        instance = ScsInstance(prob, cones)
        instance.D, instance.E = D, E
        instance.sigma, instance.rho = sigma, rho
        return instance


class ScsMultiInstance:
    """ Class that transforms a list of problems into
         a batched set of ScsInstance data.
    """

    def __init__(self, A, b, c, cones, scaled_data=None, use_tensors=True,
                 verify_sizes=False, device='cpu'):
        self.A, self.b, self.c = A, b, c
        self.all_cones = cones
        self.num_instances = len(A)
        if verify_sizes:
            raise NotImplementedError("Complete verification not implemented yet")
        if scaled_data is not None:
            D, E, sigma, rho, orig_b, orig_c = scaled_data
            self.D, self.E = D, E
            self.sigma, self.rho = sigma, rho
            self.orig_b, self.orig_c = orig_b, orig_c
            self.scaled = True
        else:

            self.scaled = False
        if use_tensors:
            self.convert_to_tensors(device)

    def get_sizes(self):
        """Get the A matrix sizes and number of instances for SCS problem"""
        assert self.num_instances > 0
        m, n = self.A[0].shape
        return m, n, self.num_instances

    def convert_to_tensors(self, device='cpu'):
        attr_list = ['A', 'b', 'c', 'D', 'E', 'sigma', 'rho', 'orig_b', 'orig_c']
        for attr in attr_list:
            if hasattr(self, attr):
                if torch.is_tensor(getattr(self, attr)):
                    continue
                value = torch.from_numpy(getattr(self, attr))
                value_device = value.to(device)
                setattr(self, attr, value_device)

    def add_solutions(self, solns):
        self.soln = solns

    def solve(self):
        solns = []
        for i in range(len(self.A)):
            # print(type(self.A[i]))
            data = {
                'A': self.A[i],
                'b': self.b[i].numpy(),
                'c': self.c[i].numpy(),
            }
            cone = self.all_cones[0]

            solns.append(scs.solve(data, cone))
        self.add_solutions(solns)
        return solns


class Problem:
    """Abstract Problem class for extracting problems in SCS format"""

    def __init__(self, config_file=None):
        self.config_file = config_file

    @abstractmethod
    def _sample_from_distributions(self, **kwargs):
        """Samples data from specified distribution to create problem instances"""
        pass

    def write_to_file(self, output_file):
        """Writes SCS format data to file."""
        pass

    def get_sizes(self, verify_sizes=False):
        """Gets the SCS instance problem sizes"""
        if hasattr(self, 'instances') is False or len(self.instances) < 1:
            raise RuntimeError("No SCS instances to get size")
        if verify_sizes:
            raise NotImplementedError("Complete verification not implemented yet")
        return self.instances[0].get_sizes()

    def get_cone(self):
        return self.instances[0].cones


class Lasso(Problem):
    """
    Constructs the Lasso problem from sampled distributions or data
    CVXPY problem format is taken from https://www.cvxpy.org/examples/machine_learning/lasso_regression.html
    Args:
        config_file: Configuration file
        n: Size of the regression variable
        n_samples: Number of problem instances that need to be generated
        train_data_size: Number of training samples required for the Lasso problem
    """

    def __init__(self, n, config_file=None, n_samples=10, train_data_size=100, create_data=True):
        super().__init__(config_file)
        print("Lasso_init")
        if create_data:
            create_scs_format = self._construct_generic(n, train_data_size)
            self.instances = self._sample_from_distributions(create_scs_format, train_data_size, n, n_samples)

    @staticmethod
    def create_sampled_object(instances):
        """Create a wrapper object around the provided instances"""
        sampled_lasso = Lasso(-1, create_data=False)
        sampled_lasso.instances = instances
        return sampled_lasso


    def _construct_generic(self, n, train_data_size):
        """ Constructs a generic creator of SCS format problems for Lasso"""
        _beta = cp.Variable(n)
        _lambd = cp.Parameter(nonneg=True)
        _X = cp.Parameter((train_data_size, n))
        _Y = cp.Parameter(train_data_size)
        objective_fn = cp.norm2(_X @ _beta - _Y) ** 2 + _lambd * cp.norm1(_beta)
        prob = cp.Problem(cp.Minimize(objective_fn))

        def create_scs_format(X_train, Y_train, lambd):
            _X.value = X_train
            _Y.value = Y_train
            _lambd.value = lambd
            scs_prob, _, _, = prob.get_problem_data(cp.SCS)
            return scs_prob

        return create_scs_format

    def _sample_from_distributions(self, create_scs_format, m, n, n_samples=10, scs_paper=True):
        """
           Samples data from specified distribution to construct Lasso problem
           instances.
        """
        instances = []
        # Only available format right now is Lasso instances
        if scs_paper:
            for i in range(n_samples):
                X, Y, lambd = self._generate_data_scs_solver_paper(m=m, n=n)
                curr_prob = create_scs_format(X, Y, lambd)
                scs_prob = ScsInstance(curr_prob)
                instances.append(scs_prob)
        else:
            raise NotImplementedError("Only scs format data implemented")
        return instances

    def _generate_data_scs_solver_paper(self, m=4, n=20, sigma=0.1,
                                        density=0.1):
        """Generates data for Lasso regression solver as used by the SCS paper."""
        beta_star = np.random.randn(n)
        idxs = np.random.choice(range(n), int((1 - density) * n), replace=False)
        for idx in idxs:
            beta_star[idx] = 0
        X = np.random.randn(m, n)
        Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
        lambd = 0.1 * np.linalg.norm((X.transpose() @ Y), np.inf)
        return X, Y, lambd

    def mse(X, Y, beta):
        """Computes the mean-square error for input X, Y, beta"""
        loss = np.linalg.norm((X @ beta - Y), 2) ** 2
        return (1.0 / X.shape[0]) * loss


class RobustPCA(Problem):
    """
    Constructs the RobustPCA problem from sampled distributions or data
    CVXPY problem format is taken from https://www.cvxpy.org/examples/machine_learning/lasso_regression.html
    Args:
        config_file: Configuration file
        n: Size of the regression variable
        n_samples: Number of problem instances that need to be generated
        train_data_size: Number of training samples required for the RobustPCA problem
    """

    def __init__(self, p=30, config_file=None, n_samples=10, q=3, create_data=True):
        super().__init__(config_file)
        print("PCA_init")
        if create_data:
            create_scs_format = self._construct_generic(p, q)
            self.instances = self._sample_from_distributions(create_scs_format, p, q, n_samples)

    @staticmethod
    def create_sampled_object(instances):
        """Create a wrapper object around the provided instances"""
        sampled_rpca = RobustPCA(-1, create_data=False)
        sampled_rpca.instances = instances
        return sampled_rpca

    def _construct_generic(self, p, q):
        """ Constructs a generic creator of SCS format problems for RobustPCA"""
        _L = cp.Variable([p, q])
        _S = cp.Variable([p, q])
        _mu = cp.Parameter(nonneg=True)
        _M = cp.Parameter([p, q])
        # objective_fn = cp.norm2(_X @ _beta - _Y) ** 2 + _lambd * cp.norm1(_beta)
        objective_fn = cp.normNuc(_L)
        constraints = [
            cp.norm1(_S) <= _mu,
            _L + _S == _M
        ]
        prob = cp.Problem(cp.Minimize(objective_fn), constraints)

        def create_scs_format(M_train, mu):
            _M.value = M_train
            _mu.value = mu
            scs_prob, _, _, = prob.get_problem_data(cp.SCS)
            return scs_prob

        return create_scs_format

    def _sample_from_distributions(self, create_scs_format, p, q, n_samples=10, scs_paper=True):
        """
           Samples data from specified distribution to construct RobustPCA problem
           instances.
        """
        instances = []
        # Only available format right now is Lasso instances
        if scs_paper:
            for i in range(n_samples):
                # X, Y, lambd = self._generate_data_scs_solver_paper(m=m, n=
                M, mu = self._generate_data_scs_solver_paper(p=p, q=q)
                curr_prob = create_scs_format(M, mu)
                scs_prob = ScsInstance(curr_prob)
                instances.append(scs_prob)
        else:
            raise NotImplementedError("Only scs format data implemented")
        return instances

    def _generate_data_scs_solver_paper(self, p=30, q=3, r=2,
                                        density=0.1):
        """Generates data for Lasso regression solver as used by the SCS paper."""
        # beta_star = np.random.randn(q)
        # idxs = np.random.choice(range(p*q), int((1 - density) * q), replace=False)
        # for idx in idxs:
        #     beta_star[idx] = 0
        # X = np.random.randn(m, n)
        # Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
        # lambd = 0.1 * np.linalg.norm((X.transpose() @ Y), np.inf)
        L_star = make_low_rank_matrix(n_samples=p, n_features=q, effective_rank=r)
        S_star = np.random.randn(p, q)
        idxs = np.random.choice(range(p * q), int((1 - density) * p * q), replace=False)
        for idx in idxs:
            S_star[idx//q][idx%q] = 0
        mu = numpy.linalg.norm(S_star,ord=1)
        M = L_star + S_star
        return M, mu

    # def mse(X, Y, beta):
    #     """Computes the mean-square error for input X, Y, beta"""
    #     loss = np.linalg.norm((X @ beta - Y), 2) ** 2
    #     return (1.0 / X.shape[0]) * loss


class YuanmingShi(Problem):
    """
    Args:
        config_file: Configuration file
        n: Size of the regression variable
        n_samples: Number of problem instances that need to be generated
        train_data_size: Number of training samples required for the Lasso problem
    """

    def __init__(self, config_file=None, create_data=True, n_samples=10, L=5, K=5, N_set=[1,1,1,1,1], Area=2):
        super().__init__(config_file)
        print("YMS_init")
        if create_data:
            create_scs_format = self._construct_generic(L, K, N_set, Area)
            self.instances = self._sample_from_distributions(create_scs_format, L, K, N_set, Area, n_samples , scs_paper=True)

    @staticmethod
    def create_sampled_object(instances):
        """Create a wrapper object around the provided instances"""
        sampled_yms = YuanmingShi(-1, create_data=False)
        sampled_yms.instances = instances
        return sampled_yms

    def _construct_generic(self, L, K, N_set, Area):
        """ Constructs a generic creator of SCS format problems for Lasso"""
        # cp.norm(X,
        N = np.sum(N_set, dtype=np.int)
        _W = cp.Variable([N, K],complex=True)
        _P_set = cp.Parameter([L, 1])
        _delta_set = cp.Parameter([K, 1])
        _r_set = cp.Parameter([K, 1])
        _H = cp.Parameter([N, K])

        objective_fn = cp.norm(_W, 'fro')
        # _L = cp.Variable([p, q])
        # _S = cp.Variable([p, q])
        # _mu = cp.Parameter(nonneg=True)
        # _M = cp.Parameter([p, q])
        # # objective_fn = cp.norm2(_X @ _beta - _Y) ** 2 + _lambd * cp.norm1(_beta)
        # objective_fn = cp.normNuc(_L)
        constraints = []
        sum = 0
        for l in range(L):
            constraints.append(cp.norm(_W[sum:sum+N_set[l], :],"fro") <= (_P_set[l])**0.5)
            sum = sum + N_set[l]
        for k in range(K):
            constraints.append(cp.norm(_H[:, k].T @ _W) <= ((1 + 1 / _r_set[k]) ** 0.5) * cp.real(_H[:, k].T @ _W[:, k]))
        prob = cp.Problem(cp.Minimize(objective_fn), constraints)
        # print(prob)
        def create_scs_format(P_set, delta_set, r_set, H):
            _P_set.value = P_set
            _delta_set.value = delta_set
            _r_set.value = r_set
            _H.value = H
            scs_prob, _, _, = prob.get_problem_data(cp.SCS)
            return scs_prob

        return create_scs_format



    def _sample_from_distributions(self, create_scs_format, L, K, N_set, Area, n_samples, scs_paper=True):
        """
           Samples data from specified distribution to construct Lasso problem
           instances.
        """
        instances = []
        # Only available format right now is Lasso instances
        if scs_paper:
            for i in range(n_samples):
                # X, Y, lambd = self._generate_data_scs_solver_paper(m=m, n=
                P_set, delta_set, r_set, H = self._generate_data_scs_solver_paper(L=L, K=K, N_set=N_set, Area=Area)
                curr_prob = create_scs_format(P_set, delta_set, r_set, H)
                scs_prob = ScsInstance(curr_prob)
                instances.append(scs_prob)
        else:
            raise NotImplementedError("Only scs format data implemented")
        return instances

    def _generate_data_scs_solver_paper(self, L=5, K=5, N_set=[1,1,1,1,1],
                                        Area=2):
        """Generates data for Lasso regression solver as used by the SCS paper."""
        delta_set = np.ones((K, 1))
        P_set = np.ones((L, 1))
        r_set = np.ones((K, 1))
        N = np.sum(N_set, dtype=np.int)
        LC = 1
        Q = 5
        H = np.ones((N, K)) + 1j * np.ones((N, K))
        H_hat = np.ones((N, K, LC))
        for ss in range(LC):
            H_hat[:, :, ss] = self.channel_realization(L, K, N_set, Area)
        for lp in range(LC):
            H = H_hat[:, :, lp]
            r_set = 10 ** (Q / 10) * np.ones((K, 1))
        # print(P_set.shape)
        # print(delta_set.shape)
        # print(r_set.shape)
        # print(H.shape)
        return P_set, delta_set, r_set, H
        # beta_star = np.random.randn(q)
        # idxs = np.random.choice(range(p*q), int((1 - density) * q), replace=False)
        # for idx in idxs:
        #     beta_star[idx] = 0
        # X = np.random.randn(m, n)
        # Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
        # lambd = 0.1 * np.linalg.norm((X.transpose() @ Y), np.inf)
        # L_star = make_low_rank_matrix(n_samples=p, n_features=q, effective_rank=r)
        # S_star = np.random.randn(p, q)
        # idxs = np.random.choice(range(p * q), int((1 - density) * p * q), replace=False)
        # for idx in idxs:
        #     S_star[idx // q][idx % q] = 0
        # mu = numpy.linalg.norm(S_star, ord=1)
        # M = L_star + S_star
        # return M, mu

    def channel_realization(self, L, K, N_set, Area):
        N = np.sum(N_set, dtype=np.int)
        H = np.zeros((N, K))
        U_position = Area * (numpy.random.rand(2, K) - 0.5)
        B_position = Area * (numpy.random.rand(2, K) - 0.5)
        D = np.zeros((L, K))
        for k in range(K):
            for l in range(L):
                d = (np.linalg.norm(B_position[:, l]  - U_position[:, k]) + 10)
                D[l, k] = 4.4 * (10 ** 5) / ((d ** 1.88) * 10 ** (np.random.normal(0, 6.3) / 20))
        for k in range(K):
            sum = 0
            for l in range(L):
                temp = np.random.normal(0, 1 / (2 ** 0.5), size=(N_set[l][0], 1))
                H[sum:sum + N_set[l][0], k:k] = D[l, k] * temp + temp * 1j
        # print(H.shape)
        return H

class Beamforming(Problem):
    """
    Args:
        config_file: Configuration file
        n: Size of the regression variable
        n_samples: Number of problem instances that need to be generated
        train_data_size: Number of training samples required for the Lasso problem
    """

    def __init__(self, config_file=None, create_data=True, n_samples=10, K=50, n=3, gamma = -13):
        super().__init__(config_file)
        print("Beamforming_init")
        if create_data:
            create_scs_format = self._construct_generic(K, n, gamma)
            self.instances = self._sample_from_distributions(create_scs_format, K, n, gamma, n_samples, scs_paper=True)

    @staticmethod
    def create_sampled_object(instances):
        """Create a wrapper object around the provided instances"""
        sampled_yms = Beamforming(-1, create_data=False)
        sampled_yms.instances = instances
        return sampled_yms

    def _construct_generic(self, K, n, gamma):
        """ Constructs a generic creator of SCS format problems for Lasso"""
        # cp.norm(X,
        _W = cp.Variable([n, K], complex=True)
        _H = cp.Parameter([n, K], complex=True)

        objective_fn = cp.norm(_W, 'fro')
        # _L = cp.Variable([p, q])
        # _S = cp.Variable([p, q])
        # _mu = cp.Parameter(nonneg=True)
        # _M = cp.Parameter([p, q])
        # # objective_fn = cp.norm2(_X @ _beta - _Y) ** 2 + _lambd * cp.norm1(_beta)
        # objective_fn = cp.normNuc(_L)
        constraints = []
        sum = 0
        for k in range(K):
            constraints.append(cp.norm2(_H[:, k].T * _W) + 1 <= cp.real((1/(1+gamma)) ** 0.5 * _H[:, k].T * _W[:, k]))
        prob = cp.Problem(cp.Minimize(objective_fn), constraints)
        # print(prob)
        def create_scs_format(H):
            _H.value = H
            scs_prob, _, _, = prob.get_problem_data(cp.SCS)
            return scs_prob

        return create_scs_format



    def _sample_from_distributions(self, create_scs_format,  K, n, gamma, n_samples, scs_paper=True):
        """
           Samples data from specified distribution to construct Lasso problem
           instances.
        """
        instances = []
        # Only available format right now is Lasso instances
        if scs_paper:
            for i in range(n_samples):
                H = self._generate_data_scs_solver_paper(K=K, n=n, gamma=gamma)
                curr_prob = create_scs_format(H)
                scs_prob = ScsInstance(curr_prob)
                instances.append(scs_prob)
        else:
            raise NotImplementedError("Only scs format data implemented")
        return instances

    def _generate_data_scs_solver_paper(self, K, n, gamma):
        """Generates data for Lasso regression solver as used by the SCS paper."""
        H = np.zeros((n, K), dtype=np.complex)
        for i in range(n):
            for j in range(K):
                H[i, j] = np.random.normal(0, K) + 1j * np.random.normal(0, K)
        # print(P_set.shape)
        # print(delta_set.shape)
        # print(r_set.shape)
        # print(H.shape)
        return H
        # beta_star = np.random.randn(q)
        # idxs = np.random.choice(range(p*q), int((1 - density) * q), replace=False)
        # for idx in idxs:
        #     beta_star[idx] = 0
        # X = np.random.randn(m, n)
        # Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
        # lambd = 0.1 * np.linalg.norm((X.transpose() @ Y), np.inf)
        # L_star = make_low_rank_matrix(n_samples=p, n_features=q, effective_rank=r)
        # S_star = np.random.randn(p, q)
        # idxs = np.random.choice(range(p * q), int((1 - density) * p * q), replace=False)
        # for idx in idxs:
        #     S_star[idx // q][idx % q] = 0
        # mu = numpy.linalg.norm(S_star, ord=1)
        # M = L_star + S_star
        # return M, mu

class Ridge(Problem):
    """
    Constructs the Lasso problem from sampled distributions or data
    CVXPY problem format is taken from https://www.cvxpy.org/examples/machine_learning/lasso_regression.html
    Args:
        config_file: Configuration file
        n: Size of the regression variable
        n_samples: Number of problem instances that need to be generated
        train_data_size: Number of training samples required for the Lasso problem
    """

    def __init__(self, n, config_file=None, n_samples=10, train_data_size=100, create_data=True):
        super().__init__(config_file)
        print("Ridge_init")
        if create_data:
            create_scs_format = self._construct_generic(n, train_data_size)
            self.instances = self._sample_from_distributions(create_scs_format, train_data_size, n, n_samples)

    @staticmethod
    def create_sampled_object(instances):
        """Create a wrapper object around the provided instances"""
        sampled_ridge = Ridge(-1, create_data=False)
        sampled_ridge.instances = instances
        return sampled_ridge


    def _construct_generic(self, n, train_data_size):
        """ Constructs a generic creator of SCS format problems for Lasso"""
        _beta = cp.Variable(n)
        _lambd = cp.Parameter(nonneg=True)
        _X = cp.Parameter((train_data_size, n))
        _Y = cp.Parameter(train_data_size)
        objective_fn = cp.norm2(_X @ _beta - _Y) ** 2 + _lambd * cp.norm2(_beta) ** 2
        prob = cp.Problem(cp.Minimize(objective_fn))

        def create_scs_format(X_train, Y_train, lambd):
            _X.value = X_train
            _Y.value = Y_train
            _lambd.value = lambd
            scs_prob, _, _, = prob.get_problem_data(cp.SCS)
            return scs_prob

        return create_scs_format

    def _sample_from_distributions(self, create_scs_format, m, n, n_samples=10, scs_paper=True):
        """
           Samples data from specified distribution to construct Lasso problem
           instances.
        """
        instances = []
        # Only available format right now is Lasso instances
        if scs_paper:
            for i in range(n_samples):
                X, Y, lambd = self._generate_data_scs_solver_paper(m=m, n=n)
                curr_prob = create_scs_format(X, Y, lambd)
                scs_prob = ScsInstance(curr_prob)
                instances.append(scs_prob)
        else:
            raise NotImplementedError("Only scs format data implemented")
        return instances

    def _generate_data_scs_solver_paper(self, m=4, n=20, sigma=0.1,
                                        density=0.1):
        """Generates data for Lasso regression solver as used by the SCS paper."""
        beta_star = np.random.randn(n)
        idxs = np.random.choice(range(n), int((1 - density) * n), replace=False)
        for idx in idxs:
            beta_star[idx] = 0
        X = np.random.randn(m, n)
        Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
        lambd = 0.1 * np.linalg.norm((X.transpose() @ Y), np.inf)
        return X, Y, lambd

class CompSense(Problem):
    """
    Constructs the Lasso problem from sampled distributions or data
    CVXPY problem format is taken from https://www.cvxpy.org/examples/machine_learning/lasso_regression.html
    Args:
        config_file: Configuration file
        n: Size of the regression variable
        n_samples: Number of problem instances that need to be generated
        train_data_size: Number of training samples required for the Lasso problem
    """

    def __init__(self, config_file=None, d=100, m=60, s=20, n_samples=100, create_data=True):
        super().__init__(config_file)
        print("compsense_init")
        if create_data:
            create_scs_format = self._construct_generic(d, m, s)
            self.instances = self._sample_from_distributions(create_scs_format, d, m, s, n_samples)

    @staticmethod
    def create_sampled_object(instances):
        """Create a wrapper object around the provided instances"""
        sampled_csens = CompSense(-1, create_data=False)
        sampled_csens.instances = instances
        return sampled_csens

    def _construct_generic(self, d, m, s):
        """ Constructs a generic creator of SCS format problems for Lasso"""
        _x = cp.Variable(d)
        _A = cp.Parameter([m, d])
        _z = cp.Parameter(m)
        objective_fn = cp.norm(_x, 1)
        constaints = [_A @ _x == _z]
        prob = cp.Problem(cp.Minimize(objective_fn), constaints)

        def create_scs_format(A, z):
            _A.value = A
            _z.value = z
            scs_prob, _, _, = prob.get_problem_data(cp.SCS)
            return scs_prob

        return create_scs_format

    def _sample_from_distributions(self, create_scs_format, d, m, s, n_samples=10, scs_paper=True):
        """
           Samples data from specified distribution to construct Lasso problem
           instances.
        """
        instances = []
        # Only available format right now is Lasso instances
        if scs_paper:
            for i in range(n_samples):
                A, z = self._generate_data_scs_solver_paper(d=d, m=m, s=s)
                curr_prob = create_scs_format(A, z)
                scs_prob = ScsInstance(curr_prob)
                instances.append(scs_prob)
        else:
            raise NotImplementedError("Only scs format data implemented")
        return instances

    def _generate_data_scs_solver_paper(self, d=100, m=75, s=25):
        """Generates data for Lasso regression solver as used by the SCS paper."""
        x0 = np.zeros(d)
        for i in range(s):
            x0[i] = (np.random.randint(0, 1)) * 2 - 1
        np.random.shuffle(x0)
        A = [[np.random.normal() for i in range(d)] for j in range(m)]
        A = np.array(A)
        z0 = A @ x0
        return A, z0

class MatrixComp(Problem):
    """
    Constructs the Lasso problem from sampled distributions or data
    CVXPY problem format is taken from https://www.cvxpy.org/examples/machine_learning/lasso_regression.html
    Args:
        config_file: Configuration file
        n: Size of the regression variable
        n_samples: Number of problem instances that need to be generated
        train_data_size: Number of training samples required for the Lasso problem
    """

    def __init__(self, config_file=None, m=100, n=20, n_samples=100, create_data=True):
        super().__init__(config_file)
        print("matrixcomp_init")
        if create_data:
            create_scs_format = self._construct_generic(m, n)
            self.instances = self._sample_from_distributions(create_scs_format, m, n, n_samples)

    @staticmethod
    def create_sampled_object(instances):
        """Create a wrapper object around the provided instances"""
        sampled_matrc = MatrixComp(-1, create_data=False)
        sampled_matrc.instances = instances
        return sampled_matrc

    def _construct_generic(self, m, n):
        """ Constructs a generic creator of SCS format problems for Lasso"""
        _X = cp.Variable([m, n])
        _M = cp.Parameter([m, n])
        objective_fn = cp.normNuc(_X)
        constraints = []
        for i in range(m):
            for j in range(n):
                constraints += [_X[i, j]*_M[i, j] == _M[i, j] * _M[i, j]]
        prob = cp.Problem(cp.Minimize(objective_fn), constraints)

        def create_scs_format(M):
            _M.value = M
            scs_prob, _, _, = prob.get_problem_data(cp.SCS)
            return scs_prob

        return create_scs_format

    def _sample_from_distributions(self, create_scs_format, m, n, n_samples=10, scs_paper=True):
        """
           Samples data from specified distribution to construct Lasso problem
           instances.
        """
        instances = []
        # Only available format right now is Lasso instances
        if scs_paper:
            for i in range(n_samples):
                M = self._generate_data_scs_solver_paper(m=m, n=n)
                curr_prob = create_scs_format(M)
                scs_prob = ScsInstance(curr_prob)
                instances.append(scs_prob)
        else:
            raise NotImplementedError("Only scs format data implemented")
        return instances

    def _generate_data_scs_solver_paper(self, m=100, n=20):
        """Generates data for Lasso regression solver as used by the SCS paper."""
        d = int(m*n*0.2)
        M = np.zeros((m, n))
        x = np.random.randn(m*n)
        count = 0
        for i in np.random.choice(m*n, d):
            M[i // n][i % n] = x[count]
            count += 1
        return M