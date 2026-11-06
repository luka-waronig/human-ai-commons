"""
Sensitivity Analysis for WaterToC Mesa Model
Implements OAT, Sobol, and KS sensitivity analysis methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from typing import Dict, List
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from water_toc.reproducibility import stable_seed

from water_toc.model import WaterToC

class WaterToCeSensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for WaterToC model
    """
    
    def __init__(self, 
                 n_runs_per_config: int = 10,
                 max_steps: int = 100,
                 n_jobs: int = 1,
                 seed: int = 42,
                 height: int = 20,
                 width: int = 20,
                 param_bounds=None,
                 baseline_params=None):
        """
        Initialize sensitivity analyzer
        
        Args:
            n_runs_per_config: Number of runs per parameter configuration
            max_steps: Maximum steps for each model run
            n_jobs: Number of parallel jobs (None for all cores)
        """
        self.n_runs_per_config = n_runs_per_config
        self.max_steps = max_steps
        self.n_jobs = mp.cpu_count() if n_jobs is None else n_jobs
        counts = [n_runs_per_config, max_steps, self.n_jobs]
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1 for value in counts):
            raise ValueError("replications, steps, and worker count must be positive")
        self.seed = seed
        self.height, self.width = height, width
        
        # Define parameter space
        self.param_bounds = {
            'initial_humans': [5, 20],
            'initial_ai': [5, 20],
            'ai_C_allocation': [1.0, 2.0],
            'ai_D_allocation': [2.5, 4.0],
            'max_water_capacity': [5, 15],
            'water_cell_density': [0.2, 0.5],
            'theta': [1.0, 10.0]
        }
        
        # Define baseline parameters
        self.baseline_params = {
            'initial_humans': 10,
            'initial_ai': 10,
            'ai_C_allocation': 2.0,
            'ai_D_allocation': 3.0,
            'max_water_capacity': 10,
            'water_cell_density': 0.3,
            'theta': 3.0
        }
        if param_bounds is not None:
            self.param_bounds = dict(param_bounds)
        if baseline_params is not None:
            self.baseline_params.update(baseline_params)
        for name, bounds in self.param_bounds.items():
            if name not in self.baseline_params:
                raise ValueError(f"No baseline specified for {name}")
            if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid bounds for {name}: {bounds}")
        
        # Key output metrics to analyze
        self.output_metrics = [
            'final_coop_fraction',
            'final_environment_state', 
            'final_total_water',
            'avg_coop_fraction',
            'avg_environment_state',
            'water_depletion_rate',
            'stability_index'
        ]

    def run_single_simulation(self, params: Dict, seed: int = None) -> Dict:
        """
        Run a single model simulation and extract key metrics
        
        Args:
            params: Dictionary of model parameters
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary of output metrics
        """
        config = {**self.baseline_params, **params}
        for name in ("initial_humans", "initial_ai"):
            config[name] = int(np.rint(config[name]))
        # Do not turn failed simulations into apparently usable NaN records.
        model = WaterToC(
            height=self.height, width=self.width, **config,
            seed=seed, save_snapshots=False, record_spatial=False,
        )
        for _ in range(self.max_steps):
            model.step()
        data = model.datacollector.get_model_vars_dataframe()
        post = data.iloc[1:]  # Initialization is not an observation of consumption.
        return {
            'final_coop_fraction': float(post['Coop_Fraction'].iloc[-1]),
            'final_environment_state': float(post['Environment_State'].iloc[-1]),
            'final_total_water': float(post['Total_Water'].iloc[-1]),
            'avg_coop_fraction': float(post['Coop_Fraction'].mean()),
            'avg_environment_state': float(post['Environment_State'].mean()),
            'water_depletion_rate': float((data['Total_Water'].iloc[0] - data['Total_Water'].iloc[-1]) / self.max_steps),
            # Descriptive inverse dispersion, not proof of dynamical stability.
            'stability_index': float(1.0 / (1.0 + post['Coop_Fraction'].std(ddof=0))),
        }

    def _evaluate_job(self, job):
        config_id, run_id, params = job
        seed = stable_seed("sensitivity-replicate", run_id, base_seed=self.seed)
        return {**params, **self.run_single_simulation(params, seed),
                'config_id': config_id, 'run_id': run_id, 'seed': seed}

    def run_batch_simulations(self, param_configs: List[Dict]) -> pd.DataFrame:
        """
        Run multiple simulations with different parameter configurations
        
        Args:
            param_configs: List of parameter dictionaries
            
        Returns:
            DataFrame with results
        """
        jobs = [(i, run, params) for i, params in enumerate(param_configs)
                for run in range(self.n_runs_per_config)]
        if self.n_jobs == 1:
            results = list(map(self._evaluate_job, jobs))
        else:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                results = list(pool.map(self._evaluate_job, jobs))
        return pd.DataFrame(results)

    def oat_sensitivity_analysis(self, n_samples: int = 10) -> Dict:
        """
        One-at-a-Time sensitivity analysis
        
        Args:
            n_samples: Number of samples for each parameter
            
        Returns:
            Dictionary with OAT results
        """
        print("Running OAT Sensitivity Analysis...")
        
        oat_results = {}
        
        for param_name in self.param_bounds.keys():
            print(f"Analyzing parameter: {param_name}")
            
            # Create parameter configurations
            param_configs = []
            param_values = np.linspace(
                self.param_bounds[param_name][0],
                self.param_bounds[param_name][1],
                n_samples
            )
            
            for value in param_values:
                config = self.baseline_params.copy()
                config[param_name] = value
                param_configs.append(config)
            
            # Run simulations
            results_df = self.run_batch_simulations(param_configs)
            
            # Calculate sensitivity metrics
            param_effects = {}
            for metric in self.output_metrics:
                if metric in results_df.columns:
                    # Group by parameter value and calculate mean
                    grouped = results_df.groupby(param_name)[metric].mean()
                    
                    # Calculate sensitivity measures
                    param_effects[metric] = {
                        'range': grouped.max() - grouped.min(),
                        'std': grouped.std(),
                        'correlation': results_df[param_name].corr(results_df[metric]),
                        'values': grouped.to_dict()
                    }
            
            oat_results[param_name] = {
                'effects': param_effects,
                'data': results_df
            }
        
        return oat_results

    def sobol_sensitivity_analysis(self, n_samples: int = 128) -> Dict:
        """
        Sobol global sensitivity analysis
        
        Args:
            n_samples: Number of samples for Sobol analysis
            
        Returns:
            Dictionary with Sobol indices
        """
        print("Running Sobol Sensitivity Analysis...")
        if n_samples < 2 or n_samples & (n_samples - 1):
            raise ValueError("Sobol base sample count must be a power of two >= 2")
        
        # Define problem for SALib
        problem = {
            'num_vars': len(self.param_bounds),
            'names': list(self.param_bounds.keys()),
            'bounds': list(self.param_bounds.values())
        }
        
        # Generate samples
        param_values = sobol.sample(problem, n_samples, seed=self.seed)
        
        # Create parameter configurations
        param_configs = []
        for values in param_values:
            config = dict(zip(problem['names'], values))
            param_configs.append(config)
        
        # Run simulations
        results_df = self.run_batch_simulations(param_configs)
        # SALib requires exactly one output per sampling-design row, in order.
        # Replicates estimate E[Y | parameters]; flattening them breaks the design.
        means = results_df.groupby('config_id')[self.output_metrics].mean().reindex(range(len(param_configs)))
        within_variances = results_df.groupby('config_id')[self.output_metrics].var(ddof=1).reindex(means.index)
        counts = results_df.groupby('config_id').size().reindex(means.index, fill_value=0)
        if (not np.isfinite(results_df[self.output_metrics].to_numpy()).all()
                or not np.isfinite(means.to_numpy()).all()
                or not (counts == self.n_runs_per_config).all()):
            raise ValueError("Incomplete or non-finite simulation output; Sobol analysis aborted")
        
        # Calculate Sobol indices for each output metric
        sobol_results = {}
        
        for metric in self.output_metrics:
            if metric in results_df.columns and not results_df[metric].isna().all():
                y = means[metric].to_numpy()
                if np.var(y) == 0:
                    # Variance-based indices are undefined for a constant output.
                    sobol_results[metric] = None
                else:
                    Si = sobol_analyze.analyze(problem, y, print_to_console=False, seed=self.seed)
                    sobol_results[metric] = {
                        'S1': Si['S1'],  # First-order indices
                        'ST': Si['ST'],  # Total-order indices
                        'S2': Si['S2'] if 'S2' in Si else None,  # Second-order indices
                        'S1_conf': Si['S1_conf'],  # Confidence intervals
                        'ST_conf': Si['ST_conf']
                    }
        
        return {
            'sobol_indices': sobol_results,
            'problem': problem,
            'data': results_df,
            'design': param_values,
            'conditional_means': means,
            'within_config_variances': within_variances,
            'estimand': 'Sobol indices of the replicate-mean response; check convergence with more replications',
        }

    def ks_sensitivity_analysis(self, n_samples: int = 100, n_perturbations: int = 50) -> Dict:
        """
        Kolmogorov-Smirnov sensitivity analysis
        
        Args:
            n_samples: Number of baseline samples
            n_perturbations: Number of perturbations per parameter
            
        Returns:
            Dictionary with KS test results
        """
        print("Running KS Sensitivity Analysis...")
        
        if n_samples < 2 or n_perturbations < 2:
            raise ValueError("KS sample counts must be at least two")
        rng = np.random.default_rng(self.seed)
        # Draw a common background ensemble. For each parameter, compare the
        # ensemble with that parameter fixed to its baseline (same other inputs).
        baseline_configs = []
        for _ in range(n_samples):
            config = {}
            for param, bounds in self.param_bounds.items():
                config[param] = rng.uniform(bounds[0], bounds[1])
            baseline_configs.append(config)
        
        baseline_results = self.run_batch_simulations(baseline_configs)
        
        ks_results = {}
        
        for param_name in self.param_bounds.keys():
            print(f"KS analysis for parameter: {param_name}")
            
            perturbed_configs = [{**config, param_name: self.baseline_params[param_name]}
                                 for config in baseline_configs]
            
            perturbed_results = self.run_batch_simulations(perturbed_configs)
            
            # Perform KS tests for each output metric
            param_ks_results = {}
            for metric in self.output_metrics:
                if metric in baseline_results.columns and metric in perturbed_results.columns:
                    baseline_values = baseline_results[metric].dropna()
                    perturbed_values = perturbed_results[metric].dropna()
                    
                    if len(baseline_values) > 0 and len(perturbed_values) > 0:
                        # Average stochastic replications first. The paired ensemble
                        # supplies a descriptive distance, not an iid-test p-value.
                        base_means = baseline_results.groupby('config_id')[metric].mean()
                        fixed_means = perturbed_results.groupby('config_id')[metric].mean()
                        ks_stat = stats.ks_2samp(base_means, fixed_means).statistic
                        param_ks_results[metric] = {
                            'ks_statistic': ks_stat,
                            'p_value': np.nan,
                            'significant': None,
                            'interpretation': 'descriptive distance: all inputs varied versus this input fixed',
                        }
            
            ks_results[param_name] = {
                'ks_tests': param_ks_results,
                'baseline_data': baseline_results,
                'perturbed_data': perturbed_results,
                'n_perturbations': 'deprecated: matched background uses n_samples configurations',
            }
        
        return ks_results

    def plot_oat_results(self, oat_results: Dict, save_path: str = None):
        """
        Plot OAT sensitivity analysis results
        """
        n_params = len(oat_results)
        n_metrics = len(self.output_metrics)
        
        fig, axes = plt.subplots(n_metrics, n_params, figsize=(4*n_params, 3*n_metrics))
        
        if n_metrics == 1:
            axes = axes.reshape(1, -1)
        if n_params == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric in enumerate(self.output_metrics):
            for j, (param_name, results) in enumerate(oat_results.items()):
                ax = axes[i, j]
                
                if metric in results['effects']:
                    data = results['data']
                    # Plot mean and one standard deviation (not a confidence interval).
                    grouped = data.groupby(param_name)[metric].agg(['mean', 'std'])
                    
                    ax.plot(grouped.index, grouped['mean'], 'o-', linewidth=2)
                    ax.fill_between(
                        grouped.index,
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.3
                    )
                    
                ax.set_xlabel(param_name)
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_sobol_results(self, sobol_results: Dict, save_path: str = None):
        """
        Plot Sobol sensitivity analysis results
        """
        sobol_indices = sobol_results['sobol_indices']
        problem = sobol_results['problem']
        
        # Create heatmap of first-order and total-order indices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # First-order indices
        s1_data = []
        st_data = []
        metrics_with_data = []
        
        for metric in self.output_metrics:
            if metric in sobol_indices and sobol_indices[metric] is not None:
                s1_data.append(sobol_indices[metric]['S1'])
                st_data.append(sobol_indices[metric]['ST'])
                metrics_with_data.append(metric)
        
        if s1_data:
            s1_df = pd.DataFrame(s1_data, 
                               columns=problem['names'], 
                               index=metrics_with_data)
            st_df = pd.DataFrame(st_data, 
                               columns=problem['names'], 
                               index=metrics_with_data)
            
            sns.heatmap(s1_df, annot=True, cmap='YlOrRd', ax=ax1, 
                       cbar_kws={'label': 'First-order Index'})
            ax1.set_title('First-order Sobol Indices')
            
            sns.heatmap(st_df, annot=True, cmap='YlOrRd', ax=ax2,
                       cbar_kws={'label': 'Total-order Index'})
            ax2.set_title('Total-order Sobol Indices')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_ks_results(self, ks_results: Dict, save_path: str = None):
        """
        Plot KS sensitivity analysis results
        """
        # Create heatmap of KS statistics
        ks_matrix = []
        param_names = []
        
        for param_name, results in ks_results.items():
            param_names.append(param_name)
            ks_row = []
            for metric in self.output_metrics:
                if metric in results['ks_tests']:
                    ks_row.append(results['ks_tests'][metric]['ks_statistic'])
                else:
                    ks_row.append(0)
            ks_matrix.append(ks_row)
        
        ks_df = pd.DataFrame(ks_matrix, 
                           columns=self.output_metrics, 
                           index=param_names)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(ks_df, annot=True, cmap='YlOrRd', 
                   cbar_kws={'label': 'KS Statistic'})
        plt.title('Kolmogorov-Smirnov Sensitivity Analysis')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_sensitivity_report(self, oat_results: Dict, sobol_results: Dict, 
                                  ks_results: Dict) -> pd.DataFrame:
        """
        Generate comprehensive sensitivity report
        """
        # Keep estimands and output units separate: one row per parameter/metric.
        rows = []
        names = sobol_results['problem']['names']
        for name in self.param_bounds:
            for metric in self.output_metrics:
                oat = oat_results.get(name, {}).get('effects', {}).get(metric, {})
                si = sobol_results['sobol_indices'].get(metric)
                ks = ks_results.get(name, {}).get('ks_tests', {}).get(metric, {})
                row = {
                    'Parameter': name, 'Metric': metric,
                    'OAT_Range': oat.get('range', np.nan),
                    'OAT_Correlation': oat.get('correlation', np.nan),
                    'Sobol_S1': np.nan, 'Sobol_ST': np.nan,
                    'KS_Distance': ks.get('ks_statistic', np.nan),
                }
                if si is not None and name in names:
                    index = names.index(name)
                    row.update(Sobol_S1=si['S1'][index], Sobol_ST=si['ST'][index])
                rows.append(row)
        return pd.DataFrame(rows)

# Example usage functions
def run_full_sensitivity_analysis():
    """
    Run complete sensitivity analysis suite
    """
    analyzer = WaterToCeSensitivityAnalyzer(
        n_runs_per_config=5,
        max_steps=50,  # Reduced for faster execution
        n_jobs=4
    )
    
    # Run all three types of analysis
    oat_results = analyzer.oat_sensitivity_analysis(n_samples=8)
    sobol_results = analyzer.sobol_sensitivity_analysis(n_samples=128)
    ks_results = analyzer.ks_sensitivity_analysis(n_samples=50, n_perturbations=30)
    
    # Generate plots
    analyzer.plot_oat_results(oat_results, 'oat_sensitivity.png')
    analyzer.plot_sobol_results(sobol_results, 'sobol_sensitivity.png')
    analyzer.plot_ks_results(ks_results, 'ks_sensitivity.png')
    
    # Generate report
    report = analyzer.generate_sensitivity_report(oat_results, sobol_results, ks_results)
    
    return {
        'oat_results': oat_results,
        'sobol_results': sobol_results,
        'ks_results': ks_results,
        'report': report,
        'analyzer': analyzer
    }

if __name__ == "__main__":
    # Run example analysis
    results = run_full_sensitivity_analysis()
    print("\nSensitivity Analysis Report:")
    print(results['report'])
