from experiment_manager import ExperimentManager

# Define different experiment configurations
experiment_configs = {
    'baseline': {
        'experiment_name': 'baseline',
        'episodes': 2000,
        'alpha': 0.1,        # Standard learning rate
        'gamma': 0.99,       # Discount factor
        'epsilon': 1.0,      # Initial exploration rate
        'min_epsilon': 0.01, # Minimum exploration rate
        'decay_rate': 0.995  # Standard decay rate
    },
    'fast_learning': {
        'experiment_name': 'fast_learning',
        'episodes': 2000,
        'alpha': 0.2,        # Higher learning rate
        'gamma': 0.99,
        'epsilon': 1.0,
        'min_epsilon': 0.01,
        'decay_rate': 0.99   # Faster decay
    },
    'slow_exploration': {
        'experiment_name': 'slow_exploration',
        'episodes': 2000,
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 1.0,
        'min_epsilon': 0.05,
        'decay_rate': 0.998
    },
    'fast_stable': {
        'experiment_name': 'fast_stable',
        'episodes': 2000,
        'alpha': 0.15,        # Moderate learning rate between 0.2 and 0.1
        'gamma': 0.99,
        'epsilon': 1.0,
        'min_epsilon': 0.02,  # Slightly higher minimum exploration rate
        'decay_rate': 0.992   # Slower decay rate for better stability
    },
    'fast_aggressive': {
        'experiment_name': 'fast_aggressive',
        'episodes': 2000,
        'alpha': 0.25,        # Higher learning rate for aggressive learning
        'gamma': 0.99,
        'epsilon': 1.0,
        'min_epsilon': 0.01,
        'decay_rate': 0.988   # Faster decay to reach exploitation phase sooner
    }
}

# Run experiments
manager = ExperimentManager()
results = manager.run_multiple_experiments(experiment_configs)

# Compare results
manager.compare_experiments()

# Generate and save report
summary = manager.generate_summary_report()
print("\nExperiment Summary:")
print(summary)

# Save report
summary.to_csv('experiment_summary.csv') 