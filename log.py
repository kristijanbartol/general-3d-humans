def log_stdout(dataset, epoch_idx, iteration, fidx, num_frames, global_metrics, pool_metrics):
    print(f'[{dataset}] Epoch: {epoch_idx}, Iteration: {iteration} ({fidx + 1}/{num_frames} frames), [MPJPE: {global_metrics.wavg.error:.2f}, Diff to [4-triang: {global_metrics.diff_to_triang:.2f}, Average: {global_metrics.diff_to_avg:.2f}, Random: {global_metrics.diff_to_random:.2f}], Top Error: {global_metrics.top.error:.2f}, Bottom Error: {global_metrics.bottom.error:.2f}]\n'
        f'\tBest (per) Loss: \t({pool_metrics.best.loss.item():.4f}, {pool_metrics.best.score.item():.4f})\n'
        f'\tBest (per) Score: \t({pool_metrics.top.loss.item():.4f}, {pool_metrics.top.score.item():.4f})\n'
        f'\t4-triang Loss: \t\t({pool_metrics.triang.loss.item():.4f})\n'
        f'\tWeighted Error: \t({pool_metrics.wavg.loss.item():.4f}, {pool_metrics.avg.loss.item():.4f}, {pool_metrics.random.loss.item():.4f})\n'
        f'\tPose Prior: \t\t ({global_metrics.triang.ratio_variances.left_right:.4f}, {global_metrics.wavg.ratio_variances.left_right:.4f}, {global_metrics.top.ratio_variances.left_right:.4f}, {global_metrics.avg.ratio_variances.left_right:.4f}, {global_metrics.bottom.ratio_variances.left_right:.4f})',
        flush=True
    )


def log_line():
    pass
