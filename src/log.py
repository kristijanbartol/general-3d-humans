def log_stdout(dataset, epoch_idx, iteration, fidx, num_frames, global_metrics, pool_metrics=None, detailed=False):
    if detailed and pool_metrics is not None:
        print(f'[{dataset}] Epoch: {epoch_idx}, Iteration: {iteration} ({fidx + 1}/{num_frames} frames), MPJPE: {global_metrics.wavg.error:.2f}mm' 
            f'Diff to [Average: {global_metrics.diff_to_avg:.2f}, Random: {global_metrics.diff_to_random:.2f}]'
            f'\tBest (per) Loss: \t{pool_metrics.best.loss.item():.2f}mm\n'
            f'\tBest (per) Score: \t{pool_metrics.most.loss.item():.2f}mm\n'
            f'\tWeighted Error: \t{pool_metrics.wavg.loss.item():.2f}mm)\n'
            f'\tMost Error: \t{global_metrics.most.error:.2f}mm'
            f'\tLeast Error: \t{global_metrics.least.error:.2f}mm\n',
            flush=True
        )
    else:
        print(f'[{dataset}] Epoch: {epoch_idx}, Iteration: {iteration} ({fidx + 1}/{num_frames} frames), MPJPE: {global_metrics.wavg.error:.2f}mm',
            flush=True)
