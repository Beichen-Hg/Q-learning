import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import json

class ExperimentAnalyzer:
    def __init__(self, results_dir='experiment_results'):
        self.results_dir = results_dir
        
    def load_experiments(self, pattern='**/stats.csv'):
        """Load all experiment data"""
        experiment_data = {}
        
        # Find all experiment data files
        all_stats_files = glob(os.path.join(self.results_dir, pattern))
        
        # Keep only the latest results for each experiment name
        latest_experiments = {}
        for stats_file in all_stats_files:
            exp_dir = os.path.dirname(stats_file)
            full_name = os.path.basename(exp_dir)
            
            # Extract experiment name from full directory name
            exp_parts = full_name.split('_')
            if len(exp_parts) >= 3:
                exp_name = '_'.join(exp_parts[:-1])
            else:
                exp_name = exp_parts[0]
            
            timestamp = exp_parts[-1]
            
            # Update if first result or newer than existing
            if exp_name not in latest_experiments or timestamp > latest_experiments[exp_name][1]:
                latest_experiments[exp_name] = (stats_file, timestamp)
        
        # Load latest experiment data
        for exp_name, (stats_file, _) in latest_experiments.items():
            exp_dir = os.path.dirname(stats_file)
            
            # Load data
            stats_df = pd.read_csv(stats_file)
            config_file = os.path.join(exp_dir, 'config.json')
            final_model_file = os.path.join(exp_dir, 'final_model.json')
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = None
            
            # Load final_model
            if os.path.exists(final_model_file):
                with open(final_model_file, 'r') as f:
                    final_model = json.load(f)
            else:
                final_model = {}
            
            experiment_data[exp_name] = {
                'stats': stats_df,
                'config': config,
                'final_model': final_model,
                'training_time': 0
            }
            
            print(f"Loaded latest results for {exp_name}: {os.path.basename(exp_dir)}")
        
        return experiment_data
        
    def analyze_learning_performance(self, data):
        """Analyze learning performance"""
        results = {}
        for name, exp in data.items():
            stats = exp['stats']
            results[name] = {
                'Average Score': stats['scores'].mean(),
                'Max Score': stats['scores'].max(),
                'Final Score': stats['scores'].iloc[-20:].mean(),  # Average of last 20 episodes
                'Average Steps': stats['steps'].mean(),
                'Convergence Episode': self._find_convergence(stats['scores']),
                'Learning Stability': stats['scores'].std()  # Score standard deviation
            }
        return pd.DataFrame(results).T
        
    def _find_convergence(self, scores, window=50, threshold=0.1):
        """Find convergence point (first point where score change rate is below threshold)"""
        # More reasonable convergence criteria
        rolling_mean = scores.rolling(window=window).mean()
        rolling_std = scores.rolling(window=window).std()
        
        # Use relative change rate instead of coefficient of variation
        changes = rolling_mean.diff().abs() / rolling_mean
        # Multiple consecutive periods must meet threshold to be considered convergent
        stable_periods = (changes < threshold).rolling(window=10).sum()
        
        # Find first stable point
        convergence_point = stable_periods[stable_periods >= 8].index[0] \
            if len(stable_periods[stable_periods >= 8]) > 0 else len(scores)
        
        return convergence_point
        
    def plot_learning_curves(self, data, metrics=['scores', 'steps', 'epsilons']):
        """Plot learning curves comparison"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5*n_metrics))
        if n_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            ax = axes[i]
            for name, exp in data.items():
                stats = exp['stats']
                if metric in stats.columns:
                    ax.plot(stats['episode'], stats[metric], label=name)
                    
            ax.set_title(f'{metric.capitalize()} over Episodes')
            ax.set_xlabel('Episode')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            
        plt.tight_layout()
        return fig
        
    def analyze_exploration_efficiency(self, data):
        """Analyze exploration efficiency"""
        results = {}
        for name, exp in data.items():
            stats = exp['stats']
            
            # Calculate average scores for exploration and exploitation phases
            exploration_phase = stats[stats['epsilons'] > 0.5]
            exploitation_phase = stats[stats['epsilons'] <= 0.1]
            
            results[name] = {
                'Exploration Phase Avg': exploration_phase['scores'].mean(),
                'Exploitation Phase Avg': exploitation_phase['scores'].mean(),
                'Exploration Length': len(exploration_phase),
                'Score Improvement': (exploitation_phase['scores'].mean() / exploration_phase['scores'].mean() - 1) * 100
            }
        return pd.DataFrame(results).T
        
    def analyze_exploration_phases(self, data):
        """Analyze exploration-exploitation phase transitions"""
        results = {}
        for name, exp in data.items():
            stats = exp['stats']
            config = exp['config']
            
            # Calculate episodes for different epsilon thresholds
            epsilon = config['epsilon']
            decay = config['decay_rate']
            episodes = []
            
            # Calculate episodes to reach 0.5 and 0.1
            for threshold in [0.5, 0.1]:
                episode = np.log(threshold / epsilon) / np.log(decay)
                episodes.append(int(episode))
            
            results[name] = {
                'Exploration Episodes': episodes[0],
                'Transition Episodes': episodes[1] - episodes[0],
                'Exploitation Start Episode': episodes[1],
                'Total Episodes': len(stats)
            }
        
        return pd.DataFrame(results).T
        
    def generate_report(self, data):
        """Generate comprehensive analysis report"""
        # Basic performance analysis
        performance = self.analyze_learning_performance(data)
        exploration = self.analyze_exploration_efficiency(data)
        phases = self.analyze_exploration_phases(data)
        
        # Create report
        report = f"""
Experiment Analysis Report
============

1. Learning Performance Comparison
{performance.to_string()}

2. Exploration-Exploitation Phase Analysis
{phases.to_string()}

3. Learning Efficiency Analysis
{exploration.to_string()}

4. Configuration Analysis
baseline:
- Highest final score ({performance.loc['baseline', 'Final Score']:.2f})
- Highest average score ({performance.loc['baseline', 'Average Score']:.2f})
- Higher learning variance (std: {performance.loc['baseline', 'Learning Stability']:.2f})

fast_aggressive:
- Quickest to exploitation phase ({phases.loc['fast_aggressive', 'Exploitation Start Episode']:.0f} episodes)
- Highest score improvement ({exploration.loc['fast_aggressive', 'Score Improvement']:.2f}%)
- Lower final score ({performance.loc['fast_aggressive', 'Final Score']:.2f})

slow_exploration:
- Most stable learning process (std: {performance.loc['slow_exploration', 'Learning Stability']:.2f})
- Longest exploration phase ({phases.loc['slow_exploration', 'Exploration Episodes']:.0f} episodes)
- Lowest average score ({performance.loc['slow_exploration', 'Average Score']:.2f})

5. Resource Utilization Analysis
{self._analyze_resources(data)}

6. Configuration Recommendations
Scenario recommendations:
1. For best final performance: baseline
   - Pros: Highest final score, good average performance
   - Cons: Higher learning variance, requires longer training time

2. For quick training: fast_aggressive
   - Pros: Fastest convergence, significant early improvements
   - Cons: Lower final performance, may sacrifice exploration

3. For stable learning: slow_exploration
   - Pros: Most stable learning process, suitable for long-term training
   - Cons: Slow convergence, lower average performance

7. Performance Trade-off Analysis
1. Learning Speed vs Stability:
   - fast_aggressive shows fastest learning but sacrifices final performance
   - slow_exploration provides most stable learning but requires longer training
   - baseline achieves good balance between speed and stability

2. Exploration vs Exploitation:
   - Shorter exploration phase (fast_aggressive) leads to quick but potentially incomplete learning
   - Longer exploration phase (slow_exploration) provides more thorough but slower learning
   - Medium exploration time (baseline) balances learning completeness and efficiency
"""
        return report

    def _analyze_resources(self, data):
        """Analyze resource utilization"""
        resource_analysis = {}
        for name, exp in data.items():
            stats = exp['stats']
            q_table_size = len(exp['final_model'])
            
            resource_analysis[name] = {
                'Q-table Size': q_table_size,
                'Training Time': exp['training_time'],
                'Memory Usage': f"~{q_table_size * 8 / 1024:.1f}MB"
            }
        
        return pd.DataFrame(resource_analysis).T.to_string()

    def save_report(self, report, filename='analysis_report.txt'):
        """Save analysis report"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

    def generate_summary_report(self):
        """Generate experiment summary report"""
        summary = []
        for name, result in self.experiments.items():
            summary.append({
                'Experiment': name,
                'Average Score': np.mean(result['stats']['scores']),
                'Max Score': max(result['stats']['scores']),
                'Final Win Rate': result['stats']['win_rate'][-1],
                'Training Time': result['training_time'],
                'Config': str(result['config'])
            })
        
        summary_df = pd.DataFrame(summary)
        return summary_df

# Example usage
if __name__ == "__main__":
    analyzer = ExperimentAnalyzer()
    data = analyzer.load_experiments()
    
    # Print found experiments
    print("\nFound experiments:")
    for name in data.keys():
        print(f"- {name}")
    
    # Generate analysis plots
    fig = analyzer.plot_learning_curves(data)
    fig.savefig('learning_analysis.png')
    
    # Generate analysis report
    report = analyzer.generate_report(data)
    analyzer.save_report(report)
    
    print("\nAnalysis complete! Please check analysis_report.txt and learning_analysis.png")

def load_experiment_data(experiment_dir):
    """Load experiment data"""
    # Load configuration
    with open(os.path.join(experiment_dir, 'config.json'), 'r') as f:
        config = json.load(f)
        
    # Load statistics
    stats = pd.read_csv(os.path.join(experiment_dir, 'stats.csv'))
    
    # Load final_model
    with open(os.path.join(experiment_dir, 'final_model.json'), 'r') as f:
        final_model = json.load(f)
        
    return {
        'config': config,
        'stats': stats.to_dict('list'),
        'final_model': final_model
    }