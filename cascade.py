import argparse
import time
import librosa
from tqdm import tqdm
import os
import glob
import torch
import numpy as np
import torch.nn as nn
from utils import demix_track, demix_track_demucs, get_model_from_config
from df.enhance import enhance, init_df, save_audio
from concurrent.futures import ThreadPoolExecutor
import concurrent
import math
import torchaudio

import warnings
warnings.filterwarnings("ignore")



def pad_audio_with_silence(audio, sr, pad_duration=3):
    """Pads an audio tensor loaded from torchaudio with silence on both sides.

    Args:
        audio (torch.Tensor or np.ndarray): Audio data.
        sr (int): Sample rate of the audio.
        pad_duration (float, optional): Duration of silence padding in seconds. Defaults to 3.

    Returns:
        torch.Tensor: Padded audio tensor with silence.
    """
    try:
        # Check if audio is a NumPy array and convert it to a PyTorch tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        elif not isinstance(audio, torch.Tensor):
            raise TypeError("audio must be either a NumPy array or a PyTorch tensor")

        # Check if sample rate is a positive integer
        if not isinstance(sr, int) or sr <= 0:
            raise ValueError("Sample rate (sr) must be a positive integer")

        # Check if pad_duration is a non-negative number
        if not isinstance(pad_duration, (int, float)) or pad_duration < 0:
            raise ValueError("pad_duration must be a non-negative number")

        # Calculate number of samples for padding based on sample rate and duration
        pad_length = math.ceil(pad_duration * sr)

        # Create silence tensor with the same data type and number of channels as audio
        if audio.ndim == 1:
            silence = torch.zeros(pad_length, dtype=audio.dtype)
        elif audio.ndim == 2:
            silence = torch.zeros((pad_length, audio.shape[1]), dtype=audio.dtype)
        else:
            raise ValueError("audio must be a 1D or 2D tensor")

        # Pad the audio with silence on both sides (before and after)
        padded_audio = torch.cat((silence, audio, silence), dim=0)

        return padded_audio
    
    except Exception as e:
        print(f"Padding Error: {e}")
        return None

def process_audio(model, args, config, device, verbose=False, enhance_model=None, df_state=None, path=None):
    try:
        mix, sr = librosa.load(path, sr=df_state.sr(), mono=False)
        mix=mix.T
        
        # mono to stereo
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=-1)
 
        mix = pad_audio_with_silence(mix, sr, pad_duration=2)

        mixture = torch.tensor(mix.T, dtype=torch.float32)

        if args.model_type == 'htdemucs':
            res = demix_track_demucs(config, model, mixture, device)
        else:
            res = demix_track(config, model, mixture, device)

        vocals = torch.from_numpy(res["vocals"])
        enhanced = enhance(enhance_model, df_state, vocals)
        enhanced=torch.mean(enhanced,dim=0,keepdim=True)
        save_path = f"{args.store_dir}/{os.path.basename(path)[:-4]}.wav" 
        torchaudio.save(uri=save_path,src=enhanced,sample_rate=sr,format="wav")
        # save_audio(save_path, enhanced, sr)
    except FileNotFoundError:
        print(f"Denoise Error: File not found at path {path}")
    except ValueError as ve:
        print(f"Denoise ValueError: {ve}")
    except TypeError as te:
        print(f"Denoise TypeError: {te}")
    except Exception as e:
        print(f"Denoise An unexpected error occurred: {e}")

def run_folder(model, args, config, device, verbose=False, enhance_model=None, df_state=None, n_workers=4):
    try:
        start_time = time.time()
        model.eval()
        all_mixtures_path = glob.glob(f'{args.input_folder}/*.wav')
        print('Total files found: {}'.format(len(all_mixtures_path)))

        if not os.path.isdir(args.store_dir):
            os.mkdir(args.store_dir)

        # Check if there are any audio files to process
        if not all_mixtures_path:
            raise FileNotFoundError(f"No audio files found in the directory: {args.input_folder}")

        # Process the first audio file to ensure the setup works
        process_audio(model, args, config, device, verbose, enhance_model, df_state, all_mixtures_path[0])

        # Process remaining audio files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_audio, model, args, config, device, verbose, enhance_model, df_state, item) for item in all_mixtures_path]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),ascii="░▒█"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file: {e}")

        time.sleep(1)
        print("Elapsed time: {:.2f} sec".format(time.time() - start_time))

    except FileNotFoundError as fnf_error:
        print(f"FileNotFoundError: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", default="", type=str, help="path to store results as wav file")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of CPUs")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true', help="invert vocals to get instrumental if provided")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    enhance_model, df_state, _ = init_df()
    num_workers=args.num_workers
    if args.start_check_point != '':
        print('Start from checkpoint: {}'.format(args.start_check_point))
        state_dict = torch.load(args.start_check_point, map_location=torch.device("cpu"))
        if args.model_type == 'htdemucs':
            # Fix for htdemucs pround etrained models
            if 'state' in state_dict:
                state_dict = state_dict['state']
        model.load_state_dict(state_dict)
    print("Instruments: {}".format(config.training.instruments))

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids)==int:
            device = torch.device(f'cuda:{device_ids}')
            model = model.to(device)
            enhance_model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = torch.device('cpu') 
        print('CUDA is not available. Running inference on CPU. It will be very slow..')
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False, enhance_model=enhance_model, df_state=df_state,n_workers=num_workers)


if __name__ == "__main__":
    proc_folder(None)