import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import pickle
import warnings
import time

from models.gpvae_ball import GPVAEBall_SWS, GPVAEBall_VNN
from utils.moving_ball import generate_moving_ball, generate_video_dict, plot_balls


def run_moving_ball(args):
    current_time = datetime.now().strftime("%m%d_%H_%M_%S")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
    torch.set_default_dtype(torch.float64 if args.float64 else torch.float32)
    if args.float64:
        warnings.warn('Faiss does not support torch.float64 now.')

    print(f"Time: {current_time}, Model: {args.model_type}-H{args.H}, Using: {device}, seed: {args.seed}, "
          f"{torch.get_default_dtype()}\n")

    s = time.time()
    if args.model_type == 'SWS':
        model = GPVAEBall_SWS(
            H=args.H, num_videos=args.num_videos, num_frames=args.num_frames, img_size=32,
            lengthscale=args.lengthscale_data, r=3, GP_joint=args.GP_joint,
            search_device=device, jitter=1e-6,   # ⚠️ jitter=1e-6 will affect KL in this SWS model
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        model.train_gpvae(
            optimizer, epochs=args.num_epochs, batch_size=None, print_epochs=args.num_print_epochs, device=device
        )

    elif args.model_type == 'VNN':
        model = GPVAEBall_VNN(
            H=args.H, num_videos=args.num_videos, num_frames=args.num_frames, img_size=32,
            lengthscale=args.lengthscale_data, r=3, GP_joint=args.GP_joint,
            search_device=device, jitter=1e-6
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        model.train_gpvae(
            optimizer, epochs=args.num_epochs, batch_size_kl=None, batch_size_expected_log=None,
            print_epochs=args.num_print_epochs, device=device
        )
    else:
        raise ValueError(f"Model type {args.model_type} not recognized")
    # model.load_state_dict(torch.load("model.pth"))

    # generate test video collection
    test_dict_coll, test_data_pickle = [], []
    for seed in range(args.num_epochs * 3, args.num_epochs * 3 + 10):  # guarantee the seed didn't appear in training
        p, v = generate_moving_ball(lengthscale=args.lengthscale_data, seed=seed)
        data_dict = generate_video_dict(p, v)
        test_dict_coll.append(data_dict)
        test_data_pickle.append((p, v))

    path_coll, target_path_coll, \
        rec_img_coll, target_img_coll, se_coll = model.predict_gpvae(test_dict_coll, device=device)

    print(f"\nMean SE: {np.mean(se_coll)}, Std: {np.std(se_coll)};\n"
          f" Test SE for each test batch: {se_coll}\n")

    # plot trajectories
    fig, axes = plot_balls(target_img_coll, target_path_coll, rec_img_coll, path_coll, nplots=args.num_plots)

    e = time.time()
    print(f"Total time: {e - s}\n")

    if not args.GP_joint:
        experiment_dir = Path(f'./exp_moving_ball_{args.model_type}_H{args.H}')
    else:
        experiment_dir = Path(f'./exp_moving_ball_{args.model_type}_H{args.H}_GP_joint')
    if args.save:
        experiment_dir.mkdir(exist_ok=True)
        trial_dir = experiment_dir/f'{current_time}'
        trial_dir.mkdir()

        torch.save(model.state_dict(), trial_dir/'model_ball.pth')             # save model

        img_path = trial_dir/'img_ball.pdf'
        fig.savefig(img_path, bbox_inches='tight')                             # save a few images

        everything_for_imgs = {
            'path_coll': path_coll, 'target_path_coll': target_path_coll,      # [10*v,f,2]
            'rec_img_coll': rec_img_coll, 'target_img_coll': target_img_coll,  # [10*v,f,32,32]
            'se_coll': se_coll
        }
        imgs_pickle_path = trial_dir/'everything_for_imgs.pkl'                 # save everything for future plotting
        with imgs_pickle_path.open('wb') as f:
            pickle.dump(everything_for_imgs, f)

        print("The experiment results are saved at:")
        print(f"{trial_dir.resolve()}")                                        # absolute path for saving log
    else:
        print("The experimental results are not saved.")

    if args.save_test_videos:  # save videos for baseline testing
        experiment_dir.mkdir(exist_ok=True)
        test_data_pickle_path = experiment_dir/f'Test_Batches_{int(args.lengthscale_data)}_{args.num_frames}.pkl'
        with test_data_pickle_path.open('wb') as f:
            pickle.dump(test_data_pickle, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Moving Ball Experiment')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--float64', action='store_true', help='whether to use float64')

    parser.add_argument('--lengthscale_data', type=float, default=2., help='true lengthscale of the data')
    parser.add_argument('--GP_joint', action='store_true', help='whether to jointly train the GP params')
    parser.add_argument('--num_videos', type=int, default=35, help='number of videos per epoch')
    parser.add_argument('--num_frames', type=int, default=30, help='number of frames per video')

    parser.add_argument('--model_type', type=str, default='VNN', choices=['SWS', 'VNN'], help='model used')
    parser.add_argument('--H', type=int, default=10, help='number of nearest neighbors')

    parser.add_argument('--geco', action='store_false', help='whether to use GECO')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of training epochs')
    parser.add_argument('--num_print_epochs', type=int, default=1, help='number of printing epochs')
    parser.add_argument('--save', action='store_true', help='save everything into a folder')
    parser.add_argument('--num_plots', type=int, default=4, help='number of plots')
    parser.add_argument('--save_test_videos', action='store_true', help='save test video batches for baseline testing')

    args_moving_ball = parser.parse_args()

    from utils.exp_utils.memo_analysis import print_RAM_usage

    print_RAM_usage()

    run_moving_ball(args_moving_ball)




