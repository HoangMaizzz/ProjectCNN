CNN evaluation output

baseline/training_history.csv: train and validation metrics for every epoch.
baseline/run_config.json: complete baseline configuration.
baseline/split_summary.csv: class counts in train, validation, and test.
baseline/evaluation/metrics_summary.json: final test metrics.
baseline/evaluation/per_class_metrics.csv: precision, recall, and F1 for every class.
baseline/evaluation/top_confusions.csv: most frequent incorrect label pairs.
baseline/evaluation/test_predictions.npz: test images, labels, predictions, and confidence.
baseline/evaluation/shift_robustness.csv: accuracy before and after artificial shifts.
baseline/evaluation/*.png: report-ready charts and example images.
experiments/: optional sample-size, learning-rate, and padding comparisons.

No user-feedback comparison is included in these evaluation results.
