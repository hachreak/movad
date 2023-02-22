import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation


def plot_scores(pred_scores, toa, tea, n_frames, out_file):
    # background
    fig, ax = plt.subplots(1, figsize=(30, 5))
    fontsize = 25
    plt.ylim(0, 1.0)
    plt.xlim(0, n_frames+1)

    xvals = np.arange(n_frames)
    plt.plot(xvals, pred_scores, linewidth=5.0, color='r')
    plt.axhline(
        y=0.5, xmin=0, xmax=n_frames + 1,
        linewidth=3.0, color='g', linestyle='--')
    if toa >= 0 and tea >= 0:
        plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
        plt.axvline(x=tea, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
        x = [toa, tea]
        y1 = [0, 0]
        y2 = [1, 1]
        ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)

    # plt.ylabel('Probability', fontsize=fontsize)
    # plt.xlabel('Frame (FPS=30)', fontsize=fontsize)
    plt.xticks(range(0, n_frames + 1, 10), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def create_curve_video(pred_scores, toa, tea, n_frames):
    # background
    fig, ax = plt.subplots(1, figsize=(30, 5))
    fontsize = 25
    plt.ylim(0, 1.0)
    plt.xlim(0, n_frames+1)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.xlabel('Frame (FPS=10)', fontsize=fontsize)
    plt.xticks(range(0, n_frames + 1, 10), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig('tmp_curve.png')
    # draw curves
    curve_writer = animation.FFMpegFileWriter(fps=10, metadata=dict(title='Movie Test', artist='Matplotlib',comment='Movie support!', codec='h264'))
    with curve_writer.saving(fig, "tmp_curve_video.mp4", 100):
        xvals = np.arange(n_frames+1)
        pred_scores = pred_scores + [pred_scores[-1]]
        for t in range(1, n_frames+1):
            plt.plot(xvals[:(t+1)], pred_scores[:(t+1)], linewidth=5.0, color='r')
            plt.axhline(y=0.5, xmin=0, xmax=n_frames + 1, linewidth=3.0, color='g', linestyle='--')
            if toa >= 0 and tea >= 0:
                plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
                plt.axvline(x=tea, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
                x = [toa, tea]
                y1 = [0, 0]
                y2 = [1, 1]
                ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
            curve_writer.grab_frame()
    plt.close()
    # read frames
    cap = cv2.VideoCapture("tmp_curve_video.mp4")
    ret, frame = cap.read()
    curve_frames = []
    while (ret):
        curve_frames.append(frame)
        ret, frame = cap.read()
    return curve_frames


def create_result_video(cfg, frames, curve_frames, vis_file):
    display_fps = 10
    image_size = cfg.image_shape
    video_writer = cv2.VideoWriter(
        vis_file,
        cv2.VideoWriter_fourcc(*'DIVX'),
        display_fps,
        (image_size[1], image_size[0]))

    for t, frame_vis in enumerate(frames[cfg.NF:]):

        frame_vis = frame_vis.astype(np.uint8)
        frame_vis = frame_vis[..., ::-1].copy()  # rgb -> bgr
        curve_img = curve_frames[t]

        shape = curve_img.shape
        curve_height = int(shape[0] * (image_size[1] / shape[1]))
        curve_img = cv2.resize(
            curve_img, (image_size[1], curve_height),
            interpolation=cv2.INTER_AREA)

        frame_vis[image_size[0]-curve_height:image_size[0]] = cv2.addWeighted(frame_vis[image_size[0]-curve_height:image_size[0]], 0.4, curve_img, 0.6, 0)

        video_writer.write(frame_vis)
