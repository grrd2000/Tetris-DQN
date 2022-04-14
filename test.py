import argparse
import torch
import cv2
from tetris.tetris import Tetris

video = True


def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=24, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=60, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.avi")

    args = parser.parse_args()
    return args


def test(options):
    print("Cuda available: ", torch.cuda.is_available())
    '''if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/tetris_100".format(options.saved_path))
    else:
        model = torch.load("{}/tetris_100".format(options.saved_path), map_location=lambda storage, loc: storage)'''
    torch.manual_seed(123)
    model = torch.load("{}/tetris_3000".format(options.saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=options.width, height=options.height, block_size=options.block_size)
    env.reset()
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter(options.output, fourcc, options.fps,
                          (int(1.5 * options.width * options.block_size), options.height * options.block_size))
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=video, vid=out)

        if done:
            out.release()
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
