import argparse

import torch
import cv2
from tetris import Tetris

video = True


def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=24, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--max_epoch_score", type=int, default=100000, help="Maximum points per epoch")
    parser.add_argument("--fps", type=int, default=120, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="output/trained_models")
    parser.add_argument("--output", type=str, default="output/recordings/output")

    args = parser.parse_args()
    return args


def test(options):
    torch.manual_seed(777)
    print("{}/tetris_best_epoch891_score_10037".format(options.saved_path))
    model = torch.load("{}/tetris_best_epoch891_score_10037".format(options.saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=options.width, height=options.height,
                 block_size=options.block_size, maxScore=options.max_epoch_score)
    env.reset()
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter(options.output+"_epoch891_score_10037.avi", fourcc, options.fps,
                          (int(1.5 * options.width * options.block_size), options.height * options.block_size))
    while True:
        next_steps = env.get_next_states_tensor()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        print(env.step_for_tests(action, render=video, vid=out))
        _, done = env.step_for_tests(action, render=video, vid=out)

        if done:
            out.release()
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
