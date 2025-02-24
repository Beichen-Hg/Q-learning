import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import time

class ExperimentManager:
    def __init__(self):
        self.experiments = {}
        self.results_dir = 'experiment_results'
        self.curves_dir = 'training_curves'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.curves_dir, exist_ok=True)
        
    def run_experiment(self, name, config):
        """Run a single experiment"""
        from train import train
        
        start_time = time.time()
        print(f"Starting experiment: {name}")
        result = train(config)
        result['training_time'] = time.time() - start_time
        
        self.experiments[name] = result
        self._save_experiment(name, result)
        return result
        
    def run_multiple_experiments(self, configs):
        """Run multiple experiment configurations"""
        results = {}
        for name, config in configs.items():
            print(f"\nStarting experiment: {name}")
            result = self.run_experiment(name, config)
            results[name] = result
            print(f"Completed experiment: {name}")
            print(f"Training curve saved as: {result['training_curve']}")
            
        # Generate and save comparison plots
        comparison_filename = self._save_comparison_curves(results)
        print(f"\nComparison curves saved as: {comparison_filename}")
        
        # Generate summary report
        summary = self.generate_summary_report()
        self._save_summary_report(summary)
        
        return results
            
    def _save_experiment(self, name, result):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = os.path.join(self.results_dir, f"{name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save configuration and statistics
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(result['config'], f, indent=4)
            
        # Save statistics as CSV
        stats_df = pd.DataFrame(result['stats'])
        stats_df.to_csv(os.path.join(experiment_dir, 'stats.csv'))
        
        # Save final model
        with open(os.path.join(experiment_dir, 'final_model.json'), 'w') as f:
            json.dump(result['final_model'], f, indent=4)
        
    def compare_experiments(self):
        """Compare experiment results (public interface)"""
        if not self.experiments:
            print("No experiments to compare")
            return
            
        # Call internal method to generate comparison plots
        comparison_filename = self._save_comparison_curves(self.experiments)
        print(f"\nComparison curves saved as: {comparison_filename}")
        return comparison_filename
        
    def _save_comparison_curves(self, results):
        """Save experiment comparison curves (internal method)"""
        plt.figure(figsize=(15, 10))
        
        # Score comparison
        plt.subplot(2, 1, 1)
        for name, result in results.items():
            plt.plot(
                result['stats']['episode'],
                result['stats']['scores'],
                label=name
            )
        plt.title('Score Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()
        
        # Steps comparison
        plt.subplot(2, 1, 2)
        for name, result in results.items():
            plt.plot(
                result['stats']['episode'],
                result['stats']['steps'],
                label=name
            )
        plt.title('Steps Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
        
        # Save comparison plots
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'comparison_curves_{timestamp}.png'
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.curves_dir, filename))
        plt.close()
        
        return filename
        
    def generate_summary_report(self):
        """Generate experiment summary report"""
        summary = []
        for name, result in self.experiments.items():
            summary.append({
                'Experiment': name,
                'Avg Score': np.mean(result['stats']['scores']),
                'Max Score': max(result['stats']['scores']),
                'Final Win Rate': result['stats']['win_rate'][-1],
                'Training Time': result['training_time'],
                'Config': str(result['config'])
            })
        
        summary_df = pd.DataFrame(summary)
        return summary_df
        
    def _save_summary_report(self, summary_df):
        """Save summary report"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'experiment_summary_{timestamp}.csv'
        summary_df.to_csv(os.path.join(self.results_dir, filename))
        print(f"\nSummary report saved as: {filename}") 