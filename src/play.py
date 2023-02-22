import os

from tqdm import tqdm

import utils
from plot import plot_scores, create_curve_video, create_result_video


def play(cfg, testdata_loader):
    filename = utils.get_result_filename(cfg, cfg.epoch)
    results = utils.load_results(filename)

    v_dir = utils.get_visual_directory(cfg, cfg.epoch)
    curves_out_dir = os.path.join(v_dir, 'curves')
    videos_out_dir = os.path.join(v_dir, 'videos')

    if not os.path.exists(curves_out_dir):
        os.makedirs(curves_out_dir)
    if not os.path.exists(videos_out_dir):
        os.makedirs(videos_out_dir)

    fc = results['frames_counter']
    if cfg.num_videos > -1:
        fc = fc[:cfg.num_videos]

    print('output directory {}'.format(v_dir))
    for i, counter in tqdm(enumerate(fc.tolist()), total=len(fc)):
        counter -= cfg.NF

        scores = results['outputs'][i]
        toa = max(0, results['toas'][i] - cfg.NF)
        tea = results['teas'][i] - cfg.NF
        idx = int(results['idxs'][i])

        plot_dir = os.path.join(curves_out_dir, '{:04d}.png'.format(int(idx)))
        plot_scores(scores, toa, tea, counter, plot_dir)

        if not cfg.no_make_video:
            curve_frames = create_curve_video(scores, toa, tea, counter)
            frames, _ = testdata_loader.dataset[idx]
            vis_file = os.path.join(videos_out_dir, 'vis_{}.avi'.format(i))
            create_result_video(cfg, frames, curve_frames, vis_file)
