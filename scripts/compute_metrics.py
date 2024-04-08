if __name__ == "__main__":
    import numpy as np

    from fmp.datasets.fingerspelling5 import metrics

    hand = np.random.rand(21, 3)
    fp_metrics = metrics.FingerspellingMetrics()
    metrics_output = fp_metrics(hand=hand)
    print(len(metrics_output))
