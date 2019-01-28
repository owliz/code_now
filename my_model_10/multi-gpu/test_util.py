import os
import numpy as np
import scipy.io as scio

# 针对远程服务器没有GUI
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.ioff()


def load_groundtruth_from_mat(dataset, gt_root_dir):
    gt_dir = os.path.join(gt_root_dir, dataset, '{}.mat'.format(dataset))
    assert(os.path.isfile(gt_dir))
    # return gt
    abnormal_events = scio.loadmat(gt_dir, squeeze_me=True)['gt']
    # abnormal_events 三维， [[[1:3],[5:9]], ]
    # 加一维度
    if abnormal_events.ndim == 2:
        abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0],
                                                  abnormal_events.shape[1])
    return abnormal_events


def detected_regions(minima_dir, window_length=50):
    txt_list = [x for x in os.listdir(minima_dir) if x.startswith('minima')]
    video_nums = len(txt_list)
    regions = []
    assert video_nums is not None, '[!!!] video_nums is None'
    for idx in range(video_nums):
        minimas = np.loadtxt(os.path.join(minima_dir, 'minima_{:02d}.txt'.format(idx+1)), dtype=int)
        minimas.sort(axis=0)
        rows = minimas.shape[0]
        region = [[],[]]
        if rows == 0:
            regions.append(region)
            continue
        start = max(minimas[0]-window_length, 1)
        end = minimas[0] + window_length
        for i in range(1, rows):
            minima = minimas[i]
            if end >= minima - window_length:
                end = minima + window_length
            else:
                region[0].append(start)
                region[1].append(end)
                start = max(minima-window_length, 1)
                end = minima + window_length
            if i == rows - 1:
                region[0].append(start)
                region[1].append(end)
        region_np = np.array(region) -1
        regions.append(region_np)
    return regions


def event_counter(regularity_score_dir, dataset, gt_root_dir, minima_dir, start_id, window_length):
    IGNORED_FRAMES = start_id
    video_length_list = np.loadtxt(os.path.join(regularity_score_dir,
                                                'video_length_list.txt')).tolist()
    txt_list = [x for x in os.listdir(minima_dir) if x.startswith('minima')]
    video_nums = len(txt_list)

    abnormal_events = load_groundtruth_from_mat(dataset, gt_root_dir)
    assert len(abnormal_events) == video_nums, \
        'the number of groundTruth does not match inference result'
    # shape  (video_nums, 2, single_video_event_nums)
    regions = detected_regions(minima_dir, window_length)
    gt_nums = 0
    detected_nums = 0
    gt_event_counter = 0
    correct_detected = []
    detected_event_counter = 0
    false_alarm = []
    # for avenue, it's 21 (vedio numbers in test file)
    num_video = abnormal_events.shape[0]
    for i in range(num_video):
        video_length = int(video_length_list[i])

        sub_abnormal_events = abnormal_events[i]
        # avenue: abnormal_events[0] 's shape: (2, 5), 5：有五个异常
        # [[  78  392  503  868  932]
        #  [ 120  422  666  910 1101]]
        # 上下对应， e.g., [77, 119]是异常帧
        # 如果缺失一维
        if sub_abnormal_events.ndim == 1:
            # 加一维度
            sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))
        # avenue:
        # (2, 5), num_abnormal=5
        _, num_gt = sub_abnormal_events.shape

        sub_region = regions[i]
        sub_region += IGNORED_FRAMES
        _, num_detected = sub_region.shape

        gt_nums += num_gt
        detected_nums += num_detected

        detected_list = np.zeros((video_length,), dtype=np.int8)
        for j in range(num_detected):
            # final_frame = video_length - 1
            detected_region = [sub_region[0, j], min(sub_region[1, j], video_length-1)]
            detected_list[detected_region[0]:detected_region[1]+1] = 1
        for j in range(num_gt):
            gt_region = [sub_abnormal_events[0, j] - 1, sub_abnormal_events[1, j] - 1]
            over_lapped = np.sum(detected_list[gt_region[0]:gt_region[1]+1] == 1)
            over_lapped_rate = over_lapped / (gt_region[1] - gt_region[0] + 1)
            if over_lapped_rate >= 0.5:
                correct_detected.append(gt_event_counter)
            gt_event_counter += 1

        gt_list = np.zeros((video_length,), dtype=np.int8)
        for j in range(num_gt):
            # final_frame = video_length - 1
            gt_region = [sub_abnormal_events[0, j] - 1, sub_abnormal_events[1, j] - 1]
            gt_list[gt_region[0]:gt_region[1] + 1] = 1
        for j in range(num_detected):
            detected_region = [sub_region[0, j], min(sub_region[1, j], video_length-1)]
            over_lapped = np.sum(gt_list[detected_region[0]:detected_region[1] + 1] == 1)
            over_lapped_rate = over_lapped / (detected_region[1] - detected_region[0] + 1)
            if over_lapped_rate < 0.5:
                false_alarm.append(detected_event_counter)
                detected_event_counter += 1

    return gt_nums, detected_nums, correct_detected, false_alarm


def plot_event(video_nums, dataset, regularity_score_dir, error_name, logger, gt_root_dir,
               start_id, minima_dir, window_length):
    video_length_list = np.loadtxt(os.path.join(regularity_score_dir, 
        'video_length_list.txt')).tolist()
    plot_dir = os.path.join(regularity_score_dir, error_name, minima_dir, 'event_png')
    print("Plotting regularity scores and event, saved in [{}]".format(plot_dir))
    logger.info("Plotting regularity scores and event, saved in [{}]".format(plot_dir))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    gt_dir = os.path.join(gt_root_dir, dataset, '{}.mat'.format(dataset))
    assert (os.path.isfile(gt_dir))
    # return gt
    abnormal_events = scio.loadmat(gt_dir, squeeze_me=True)['gt']
    # abnormal_events 三维， [[[1:3],[5:9]], ]
    # 加一维度
    if abnormal_events.ndim == 2:
        abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0],
                                                  abnormal_events.shape[1])

    num_video = abnormal_events.shape[0]
    assert num_video == video_nums
    
    # detected regions
    regions = detected_regions(minima_dir, window_length)

    # plot
    for video_idx in range(num_video):
        video_length = int(video_length_list[video_idx])
        # shape = (1435,)
        regularity_score = np.loadtxt(os.path.join(regularity_score_dir, error_name,
                                                   'scores_{:02d}.txt'.format(video_idx + 1)))
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        # plot regularity score
        x = np.arange(start_id, start_id + regularity_score.shape[0])
        ax.plot(x, regularity_score, color='b', linewidth=2.0)
        plt.xlabel('Frame number')
        plt.ylabel('Regularity score')
        plt.ylim(0, 1)
        plt.xlim(1, regularity_score.shape[0] + 1)

        # draw minima
        minimas_idx = np.loadtxt(os.path.join(minima_dir, 'minima_{:02d}.txt'.format(video_idx + 1)),
                             dtype=int)
        minimas_idx.sort(axis=0)
        minimas_idx -= 1

        plt.plot(minimas_idx + start_id, regularity_score[minimas_idx], "o", label="min")

        # draw GT region
        sub_abnormal_events = abnormal_events[video_idx]
        if sub_abnormal_events.ndim == 1:
            # 加一维度
            sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))
        _, num_abnormal = sub_abnormal_events.shape
        
        for j in range(num_abnormal):
            start = sub_abnormal_events[0, j] - 1
            end = sub_abnormal_events[1, j]
            plt.fill_between(np.arange(start, end), 0, 1, facecolor='red', alpha=0.4)

        # draw detected region
        sub_region = regions[video_idx]
        sub_region += start_id
        _, num_detected = sub_region.shape

        for j in range(num_detected):
            start = sub_region[0, j]
            end = min(sub_region[1, j], video_length-1) + 1
            plt.fill_between(np.arange(start, end), 0, 1, facecolor='green', alpha=0.4)

        plt.savefig(os.path.join(plot_dir, 'scores_video_{:02d}.png'.format(video_idx + 1)),
                    dpi=300)
        plt.close()
    
    gt_nums, detected_nums, correct_detected, false_alarm = event_counter(regularity_score_dir,
                                                                          dataset, gt_root_dir,
                                                                          minima_dir, start_id,
                                                                          window_length)
    print('gt_nums={}, detected_nums={}'.format(gt_nums, detected_nums))
    print('correct_detected={} \n, false_alarm={}'.format(len(correct_detected),
                                                          len(false_alarm)))
    logger.info('gt_nums={}, detected_nums={}'.format(gt_nums, detected_nums))
    logger.info('correct_detected={} \n, false_alarm={}'.format(len(correct_detected),
                                                                len(false_alarm)))


def load_groundtruth(regularity_score_dir, dataset, gt_root_dir):
    ignored_frames_list = np.loadtxt(os.path.join(regularity_score_dir, 'ignored_frames_list.txt'),
                                     dtype=int).tolist()
    video_length_list  = np.loadtxt(os.path.join(regularity_score_dir, 'video_length_list.txt'),
                                    dtype=int).tolist()
    gt_dir = os.path.join(gt_root_dir, dataset, '{}.mat'.format(dataset))
    assert(os.path.isfile(gt_dir))
    # return gt
    import scipy.io as scio
    abnormal_events = scio.loadmat(gt_dir, squeeze_me=True)['gt']
    # abnormal_events 三维， [[[1:3],[5:9]], ]
    # 加一维度
    if abnormal_events.ndim == 2:
        abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0],
                                                  abnormal_events.shape[1])

    # for avenue, it's 21 (vedio numbers in test file)
    num_video = abnormal_events.shape[0]
    gt = []
    for i in range(num_video):
        IGNORED_FRAMES = ignored_frames_list[i]
        # frames under subfile
        # 一维int
        video_length = video_length_list[i]
        sub_video_gt = np.zeros((video_length,), dtype=np.int8)
        sub_abnormal_events = abnormal_events[i]
        # avenue: abnormal_events[0] 's shape: (2, 5), 5：有五个异常
        # [[  78  392  503  868  932]
        #  [ 120  422  666  910 1101]]
        # 上下对应， e.g., [77, 119]是异常帧
        # 如果缺失一维
        if sub_abnormal_events.ndim == 1:
            # 加一维度
            sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))
        # avenue:
        # (2, 5), num_abnormal=5
        _, num_abnormal = sub_abnormal_events.shape

        for j in range(num_abnormal):
            start = sub_abnormal_events[0, j] - 1
            end = sub_abnormal_events[1, j]
            # 左闭右开
            sub_video_gt[start: end] = 1
        if(IGNORED_FRAMES != 0):
            sub_video_gt = sub_video_gt[IGNORED_FRAMES:]
        gt.append(sub_video_gt)
    return gt


def plot_score(video_nums, dataset, regularity_score_dir, error_name, logger, gt_root_dir,
               start_id):
    plot_dir = os.path.join(regularity_score_dir, error_name, 'png')
    print("Plotting regularity scores, saved in [{}]".format(plot_dir))
    logger.info("Plotting regularity scores, saved in [{}]".format(plot_dir))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    gt_dir = os.path.join(gt_root_dir, dataset, '{}.mat'.format(dataset))
    assert (os.path.isfile(gt_dir))
    # return gt
    import scipy.io as scio
    abnormal_events = scio.loadmat(gt_dir, squeeze_me=True)['gt']
    # abnormal_events 三维， [[[1:3],[5:9]], ]
    # 加一维度
    if abnormal_events.ndim == 2:
        abnormal_events = abnormal_events.reshape(-1, abnormal_events.shape[0],
                                                  abnormal_events.shape[1])

    num_video = abnormal_events.shape[0]
    assert num_video == video_nums
    # plot GT
    for video_idx in range(num_video):
        # shape = (1430,)
        regularity_score = np.loadtxt(os.path.join(regularity_score_dir, error_name,
                                                   'scores_{:02d}.txt'.format(video_idx + 1)))

        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

        # plot regularity score
        ax.plot(np.arange(start_id, start_id+regularity_score.shape[0]), regularity_score,
                color='b', linewidth=2.0)
        plt.xlabel('Frame number')
        plt.ylabel('Regularity score')
        plt.ylim(0, 1)
        plt.xlim(1, regularity_score.shape[0] + 1)

        # [[  78  392  503  868  932]
        #  [ 120  422  666  910 1101]]
        sub_abnormal_events = abnormal_events[video_idx]
        if sub_abnormal_events.ndim == 1:
            # 加一维度
            sub_abnormal_events = sub_abnormal_events.reshape((sub_abnormal_events.shape[0], -1))
        _, num_abnormal = sub_abnormal_events.shape
        # num_abnormal = sub_abnormal_events.shape[1]
        for j in range(num_abnormal):
            start = sub_abnormal_events[0, j] - 1
            end = sub_abnormal_events[1, j]
            plt.fill_between(np.arange(start, end), 0, 1, facecolor='red', alpha=0.4)

        plt.savefig(os.path.join(plot_dir, 'scores_video_{:02d}.png'.format(video_idx + 1)),
                    dpi=300)
        plt.close()


def plot_heatmap(video_nums, dataset, regularity_score_dir, error_name, logger,
               start_id):
    plot_dir = os.path.join(regularity_score_dir, error_name, 'heatmap')
    print("Plotting regularity scores, saved in [{}]".format(plot_dir))
    logger.info("Plotting regularity scores, saved in [{}]".format(plot_dir))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for video_idx in range(video_nums):
        # shape = (1430,)
        losses = np.load(os.path.join(regularity_score_dir, error_name,
                                                   'scores_{:02d}.npy'.format(video_idx + 1)))

        for frame_idx in range(losses.shape()[0]):
            plt.imshow(np.squeeze(losses[frame_idx]), vmin=np.amin(losses),
                       vmax=np.amax(losses), cmap='jet')
            plt.colorbar()
            plt.savefig(os.path.join(plot_dir,
                                     '{}_err_vid{:02d}_frm{:03d}.png'.format(dataset, video_idx + 1,
                                                                             frame_idx + 1)))
            plt.clf()


def confuse_scores(video_nums, regularity_score_dir, error_name, dataset, gt_root_dir):
    gt = load_groundtruth(regularity_score_dir, dataset, gt_root_dir)
    # train
    assert len(gt) == video_nums, 'the number of groundTruth does not match inference result'

    scores = np.array([], dtype=np.float32)
    labels = np.array([], dtype=np.int8)

    for i in range(video_nums):
        regularity_score = np.loadtxt(os.path.join(regularity_score_dir, error_name,
                                                   'scores_{:02d}.txt'.format(i + 1)))
        scores = np.concatenate((scores, regularity_score[:]), axis=0)
        labels = np.concatenate((labels[:], gt[i][:]), axis=0)
        assert len(regularity_score)==len(gt[i]), 'score and gt are not equal'

    np.savetxt(os.path.join(regularity_score_dir, error_name, 'scores_all_video.txt'), scores)
    scores = 1-scores
    return scores, labels


def compute_eer(video_nums, regularity_score_dir, error_name, dataset, gt_root_dir):
    """
    eer is the point where fpr==fnr(1-tpr)
    :param video_nums:
    :param regularity_score_dir:
    :param error_name:
    :return: eer
    """
    scores, labels = confuse_scores(video_nums, regularity_score_dir, error_name, dataset,
                                    gt_root_dir)
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1-tpr
    cords = zip(fpr, fnr)
    min_dist = 999999
    for item in cords:
        item_fpr, item_fnr = item
        dist = abs(item_fpr - item_fnr)
        if dist < min_dist:
            min_dist = dist
            eer = (item_fpr + item_fnr) / 2
    return eer


def compute_auc(video_nums, regularity_score_dir, error_name, dataset, gt_root_dir):
    scores, labels = confuse_scores(video_nums, regularity_score_dir, error_name, dataset,
                                    gt_root_dir)

    from sklearn import metrics
    # pos_label=1,gt中设置异常帧为1，故异常为正样本。Label considered as positive
    # scores为分为positive_label的概率，故为异常值，越大越异常
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def compute_precision_recall_auc(video_nums, regularity_score_dir, error_name, dataset,
                                 gt_root_dir):
    scores, labels = confuse_scores(video_nums, regularity_score_dir, error_name, dataset,
                                    gt_root_dir)

    from sklearn import metrics
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=1)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc


