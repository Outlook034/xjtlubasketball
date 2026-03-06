import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_utils import sparse_depth_consistency_loss_func, smoothness_loss_func
from transforms import Transforms
from net_utils import OutlierRemoval


class CompletionTTAFastModule(nn.Module):
    def __init__(self, depth_model, args):
        super().__init__()
        self.depth_model = depth_model
        self.steps = args.tta_steps
        self.lr = args.tta_lr
        self.w_sparse = args.tta_w_sparse
        self.w_smooth = args.tta_w_smooth
        self.grad_clip = args.tta_grad_clip
        self.augmentation_prob = args.tta_aug_prob

        self.outlier_removal = OutlierRemoval(7, 1.5)
        self.train_geom = Transforms(
            random_crop_to_shape=[args.height, args.width],
            random_flip_type=args.tta_aug_flip_type,
            random_rotate_max=args.tta_aug_rotate,
            random_crop_and_pad=[-1, -1],
            random_resize_and_pad=[-1, -1],
            random_resize_and_crop=[-1, -1]
        )
        self.interpolation_modes = self.train_geom.map_interpolation_mode_names_to_enums(
            ['bilinear', 'nearest', 'nearest']
        )
        self.train_photo = Transforms(
            normalized_image_range=[0, 255],
            random_brightness=args.tta_aug_brightness,
            random_contrast=args.tta_aug_contrast,
            random_gamma=args.tta_aug_gamma,
            random_hue=[-1, -1],
            random_saturation=args.tta_aug_saturation,
            random_noise_type=args.tta_aug_noise_type,
            random_noise_spread=args.tta_aug_noise_spread
        )

        self.register_buffer("norm_mean", torch.tensor(args.tta_norm_mean).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(args.tta_norm_std).view(1, 3, 1, 1))

        # Freeze base model
        for p in self.depth_model.parameters():
            p.requires_grad = False

        self.adapt_layer = None
        self.optimizer = None

    def _normalize(self, image):
        if image.max() > 1.0:
            image = image / 255.0
        return (image - self.norm_mean) / self.norm_std

    def _ensure_adapt_layer(self, feat):
        if self.adapt_layer is not None:
            return
        if feat.dim() != 4:
            raise ValueError(f"Expected 4D feature map, got {tuple(feat.shape)}")
        c = feat.shape[1]
        self.adapt_layer = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
        ).to(feat.device)
        self.optimizer = torch.optim.Adam(self.adapt_layer.parameters(), lr=self.lr)

    def _encode(self, x):
        # Support both wrappers with .encoder and plain modules
        if hasattr(self.depth_model, "encoder"):
            return self.depth_model.encoder(x)
        raise AttributeError("depth_model must have an 'encoder' attribute")

    def _decode(self, feat):
        if hasattr(self.depth_model, "decoder"):
            out = self.depth_model.decoder(feat)
            if isinstance(out, dict):
                return out[("disp", 0)]
            return out
        raise AttributeError("depth_model must have a 'decoder' attribute")

    @torch.enable_grad()
    def adapt(self, image, sparse_depth):
        validity = (sparse_depth > 0).float()
        filtered_sparse, filtered_valid = self.outlier_removal.remove_outliers(
            sparse_depth=sparse_depth,
            validity_map=validity
        )

        geom_outputs = self.train_geom.transform(
            images_arr=[image, filtered_sparse, filtered_valid],
            intrinsics_arr=[],
            interpolation_modes=self.interpolation_modes,
            random_transform_probability=self.augmentation_prob
        )
        image_geom, sparse_geom, valid_geom = geom_outputs

        sparse_geom = F.interpolate(sparse_geom, size=image_geom.shape[2:], mode='nearest')
        valid_geom = F.interpolate(valid_geom, size=image_geom.shape[2:], mode='nearest')

        [image_photo] = self.train_photo.transform(
            images_arr=[image_geom],
            random_transform_probability=self.augmentation_prob
        )

        for _ in range(self.steps):
            norm_image = self._normalize(image_photo)

            feat = self._encode(norm_image)
            self._ensure_adapt_layer(feat)
            feat_adapted = feat + self.adapt_layer(feat)
            prediction = self._decode(feat_adapted)

            sparse_for_loss = sparse_geom
            valid_for_loss = valid_geom
            if sparse_for_loss.shape[2:] != prediction.shape[2:]:
                sparse_for_loss = F.interpolate(sparse_for_loss, size=prediction.shape[2:], mode='nearest')
                valid_for_loss = F.interpolate(valid_for_loss, size=prediction.shape[2:], mode='nearest')

            image_for_loss = image_photo
            if image_for_loss.shape[2:] != prediction.shape[2:]:
                image_for_loss = F.interpolate(image_for_loss, size=prediction.shape[2:], mode='bilinear', align_corners=False)

            loss_sparse = sparse_depth_consistency_loss_func(
                prediction,
                sparse_for_loss,
                valid_for_loss
            )
            loss_smooth = smoothness_loss_func(
                predict=prediction,
                image=image_for_loss / 255.0 if image_for_loss.max() > 1.0 else image_for_loss
            )
            total_loss = self.w_sparse * loss_sparse + self.w_smooth * loss_smooth

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.adapt_layer.parameters(), self.grad_clip)
            self.optimizer.step()

        with torch.no_grad():
            feat = self._encode(self._normalize(image))
            feat = feat + self.adapt_layer(feat)
            final_prediction = self._decode(feat)
        return final_prediction
