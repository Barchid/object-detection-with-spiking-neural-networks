import os
import tqdm
from celluloid import Camera
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional
from torchvision.transforms import InterpolationMode
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from prophesee_utils.io.psee_loader import PSEELoader

class GEN1DetectionDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T
        self.sample_size = args.sample_size
        self.quantization_size = [args.sample_size // args.T, 1, 1]
        self.h, self.w = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]
        self.data_dir = os.path.join("/datas/sandbox/Gen1Detection", self.mode)
        # save_file_name = f"gen1_{mode}_{self.sample_size//1000}_{self.quantization_size[0]/1000}ms_{self.tbin}tbin.pt"
        # save_file = os.path.join("/datas/sandbox/Gen1Detection", save_file_name)

        # if os.path.isfile(save_file):
        #     self.samples = torch.load(save_file)
        #     print("File loaded.")
        # else:
        #     data_dir = os.path.join(args.path, mode)
        #     self.samples = self.build_dataset(data_dir, save_file)
        #     torch.save(self.samples, save_file)
        #     print(f"Done! File saved as {save_file}")

        if not self._check_exists():
            data_dir = os.path.join(args.path, mode)
            self.build_dataset(data_dir, None)

        self._load_dataset()

    def _load_dataset(self):
        self.files = []
        for fil in os.listdir(self.data_dir):
            if fil.split(".")[-1] != "pt":
                continue
            fil = os.path.join(self.data_dir, fil)
            self.files.append(fil)

    def __getitem__(self, index):
        fil = self.files[index]
        sample = torch.load(fil)
        (coords, feats), target = sample
        # (coords, feats), target = self.samples[index]

        sample = torch.sparse_coo_tensor(
            coords.t(),
            feats.to(torch.float32),
            # size=(self.T, self.quantized_h, self.quantized_w, self.C)
        )
        sample = sample.coalesce().to_dense().permute(0, 3, 1, 2)

        sample = (sample > 0).to(torch.float32)

        if sample.shape[0] > self.T:
            sample = sample[: self.T, :, :, :]
        elif sample.shape[0] < self.T:
            new_sample = torch.zeros(
                (self.T, self.C, self.quantized_h, self.quantized_w)
            )
            new_sample[: sample.shape[0], :, :, :] = sample

            for i in range(sample.shape[0], self.T):
                new_sample[i, :, :, :] = sample[-1, :, :, :]

            sample = new_sample
            
        sample = functional.resize(sample, (self.quantized_h, self.quantized_w), interpolation=InterpolationMode.NEAREST)

        return sample, target

    def __len__(self):
        return len(self.files)

    def build_dataset(self, path, save_file):
        # Remove duplicates (.npy and .dat)
        files = [
            os.path.join(path, time_seq_name[:-9])
            for time_seq_name in os.listdir(path)
            if time_seq_name[-3:] == "npy"
        ]

        print("Building the Dataset")
        pbar = tqdm.tqdm(total=len(files), unit="File", unit_scale=True)
        titles = []
        samples = []
        i = 0
        for file_name in files:
            print(f"Processing {file_name}...")
            events_file = file_name + "_td.dat"
            video = PSEELoader(events_file)

            boxes_file = file_name + "_bbox.npy"
            boxes = np.load(boxes_file)
            # Rename 'ts' in 't' if needed (Prophesee GEN1)
            boxes.dtype.names = [
                dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names
            ]

            boxes_per_ts = np.split(
                boxes, np.unique(boxes["t"], return_index=True)[1][1:]
            )

            for b in boxes_per_ts:
                sample = self.create_sample(video, b)
                if sample is not None:
                    samples.append(sample)
                    titles.append(
                        os.path.join(self.data_dir, str(i).zfill(6))
                    )
                    i += 1

            # samples.extend(
            #     [
            #         sample
            #         for b in boxes_per_ts
            #         if (sample := self.create_sample(video, b)) is not None
            #     ]
            # )
            pbar.update(1)

        pbar.close()
        print(f"saving samples to {self.data_dir}")
        for sample, title in zip(samples, titles):
            torch.save(sample, title + ".pt")
        print("Done!")
        return samples

    def _check_exists(self):
        if not os.path.isdir(self.data_dir):
            return False

        cnt = 0
        for fil in os.listdir(self.data_dir):
            fil = os.path.join(self.data_dir, fil)
            if os.path.isfile(fil) and fil.split(".")[-1] == "pt":
                cnt += 1

        return cnt > 0

    def create_sample(self, video, boxes):
        ts = boxes["t"][0]
        video.seek_time(ts - self.sample_size)
        events = video.load_delta_t(self.sample_size)

        targets = self.create_targets(boxes)

        if targets["boxes"].shape[0] == 0:
            print(f"No boxes at {ts}")
            return None
        elif events.size == 0:
            print(f"No events at {ts}")
            return None
        else:
            return (self.create_data(events), targets)

    def create_targets(self, boxes):
        torch_boxes = torch.from_numpy(
            structured_to_unstructured(boxes[["x", "y", "w", "h"]], dtype=np.float32)
        )

        # keep only last instance of every object per target
        _, unique_indices = np.unique(
            np.flip(boxes["track_id"]), return_index=True
        )  # keep last unique objects
        unique_indices = np.flip(-(unique_indices + 1))
        torch_boxes = torch_boxes[[*unique_indices]]

        torch_boxes[:, 2:] += torch_boxes[:, :2]  # implicit conversion to xyxy
        torch_boxes[:, 0::2].clamp_(min=0, max=self.w)
        torch_boxes[:, 1::2].clamp_(min=0, max=self.h)

        # valid idx = width and height of GT bbox aren't 0
        valid_idx = (torch_boxes[:, 2] - torch_boxes[:, 0] != 0) & (
            torch_boxes[:, 3] - torch_boxes[:, 1] != 0
        )
        torch_boxes = torch_boxes[valid_idx, :]

        torch_labels = torch.from_numpy(boxes["class_id"]).to(torch.long)
        torch_labels = torch_labels[[*unique_indices]]
        torch_labels = torch_labels[valid_idx]

        return {"boxes": torch_boxes, "labels": torch_labels}

    def create_data(self, events):
        events["t"] -= events["t"][0]
        feats = torch.nn.functional.one_hot(
            torch.from_numpy(events["p"]).to(torch.long), self.C
        )

        coords = torch.from_numpy(
            structured_to_unstructured(events[["t", "y", "x"]], dtype=np.int32)
        )

        # Bin the events on T timesteps
        coords = torch.floor(coords / torch.tensor(self.quantization_size))
        coords[:, 1].clamp_(min=0, max=self.quantized_h - 1)
        coords[:, 2].clamp_(min=0, max=self.quantized_w - 1)

        # TBIN computations
        tbin_size = self.quantization_size[0] / self.tbin

        # get for each ts the corresponding tbin index
        tbin_coords = (events["t"] % self.quantization_size[0]) // tbin_size
        # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
        tbin_feats = ((events["p"] + 1) * (tbin_coords + 1)) - 1

        feats = torch.nn.functional.one_hot(
            torch.from_numpy(tbin_feats).to(torch.long), 2 * self.tbin
        ).to(bool)

        return coords.to(torch.int16), feats.to(bool)


def animate(spikes: torch.Tensor, targets: torch.Tensor = None):
    fig, ax = plt.subplots()
    camera = Camera(fig)
    plt.axis("off")

    for i in range(spikes.shape[0]):
        spike = spikes[i].numpy()
        frm = np.full((spike.shape[1], spike.shape[2]), 127, dtype=np.uint8)

        frm[spike[0, :, :] > 0] = 0
        frm[spike[1, :, :] > 0] = 255

        boxes = targets["boxes"]
        for boxe in boxes:
            x1 = int(boxe[0].item())
            y1 = int(boxe[1].item())
            x2 = int(boxe[2].item())
            y2 = int(boxe[3].item())
            # up
            frm[y1 : y1 + 2, x1:x2] = 0
            # left
            frm[y1:y2, x1 : x1 + 2] = 0
            # right
            frm[y1:y2, x2 : x2 + 2] = 0
            # bot
            frm[y2 : y2 + 2, x1:x2] = 0

        ax.imshow(frm, cmap="Greys")  # noqa: F841
        camera.snap()

    anim = camera.animate(interval=50)
    anim.save("examples/example.gif")
    plt.close("all")
